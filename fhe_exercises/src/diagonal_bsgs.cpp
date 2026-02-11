#include "HEAAN2/HEAAN2.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <map>

using namespace heaan;

const auto preset_id = PresetParamsId::F16Opt_Gr;
const auto device = Device::CPU;

std::vector<double> diagonalMethodBSGS(
    const std::vector<std::vector<double>>& U,
    const std::vector<double>& v,
    const ISecretKey& sk,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval
) {
    int n = U.size();
    int log_slots = sk.logDegree() - 1;
    int num_slots = 1 << log_slots;
    
    std::cout << "\n=== Diagonal Method with BSGS ===" << std::endl;
    std::cout << "  Dimension: " << n << "×" << n << std::endl;
    
    // ------------------------------------------------------------
    // 1. Choisir n₁ et n₂ tels que n = n₁ × n₂
    // ------------------------------------------------------------
    int n1 = (int)std::sqrt(n);
    int n2 = n / n1;
    while (n1 * n2 < n) n2++;
    while (n1 * n2 > n) n1--;
    
    std::cout << "  n = " << n << " = " << n1 << " × " << n2 << std::endl;
    
    // ------------------------------------------------------------
    // 2. Chiffrer le vecteur v
    // ------------------------------------------------------------
    Message<Complex> msg_v(log_slots, device);
    for (int i = 0; i < n; ++i) msg_v[i] = Complex(v[i], 0.0);
    for (int i = n; i < num_slots; ++i) msg_v[i] = Complex(0.0, 0.0);
    
    auto ptxt_v = IPlaintext::make();
    EnDecoder encoder(preset_id);
    encoder.encode(msg_v, *ptxt_v);
    
    auto ct_v = ICiphertext::make();
    EnDecryptor encryptor(preset_id);
    encryptor.encrypt(*ptxt_v, sk, *ct_v);
    
    int level_v = eval.getLevel(*ct_v);
    std::cout << "  Vecteur v chiffré, niveau: " << level_v << std::endl;
    
    // ------------------------------------------------------------
    // 3. Baby Steps : précalculer Rot_i(v) pour i = 1..n₂-1
    //    i = 0 est ct_v lui-même (pas de copie!)
    // ------------------------------------------------------------
    std::vector<Ptr<ICiphertext>> baby_steps(n2);
    
    for (int i = 1; i < n2; ++i) {
        auto it = rot_keys.find(i);
        if (it == rot_keys.end()) {
            std::cerr << "  ERREUR: Clé de rotation " << i << " non trouvée!" << std::endl;
            continue;
        }
        
        auto ct_rot = ICiphertext::make();
        eval.rot(*ct_v, i, *ct_rot, *(it->second));
        baby_steps[i] = std::move(ct_rot);
        
        if (i % 10 == 0 || i == n2-1) {
            std::cout << "    Baby step " << i << "/" << n2-1 << " fait" << std::endl;
        }
    }
    std::cout << "  Baby steps (1.." << n2-1 << ") précalculés" << std::endl;
    
    // ------------------------------------------------------------
    // 4. Giant Steps : pour chaque j = 0..n₁-1
    // ------------------------------------------------------------
    std::vector<Ptr<ICiphertext>> giant_steps(n1);
    
    for (int j = 0; j < n1; ++j) {
        if (j % 10 == 0) std::cout << "  Giant step j=" << j << "/" << n1-1 << std::endl;
        
        auto ct_gs = ICiphertext::make();
        
        // --------------------------------------------------------
        // Cas i = 0 : utiliser ct_v directement
        // --------------------------------------------------------
        Message<Complex> msg_diag0(log_slots, device);
        for (int k = 0; k < n; ++k) {
            int row = ((-j * n2 + k) % n + n) % n;
            int col = k;  // i = 0
            msg_diag0[k] = Complex(U[row][col], 0.0);
        }
        for (int k = n; k < num_slots; ++k) {
            msg_diag0[k] = Complex(0.0, 0.0);
        }
        
        auto ptxt_diag0 = IPlaintext::make();
        encoder.encode(msg_diag0, *ptxt_diag0);
        
        auto ptxt_diag0_leveled = IPlaintext::make();
        eval.levelDownTo(*ptxt_diag0, *ptxt_diag0_leveled, eval.getLevel(*ct_v));
        
        auto ct_mul0 = ICiphertext::make();
        eval.mul(*ct_v, *ptxt_diag0_leveled, *ct_mul0);
        eval.rescale(*ct_mul0, *ct_mul0);
        
        ct_gs = std::move(ct_mul0);
        
        // --------------------------------------------------------
        // Cas i = 1..n₂-1 : utiliser baby_steps[i]
        // --------------------------------------------------------
        for (int i = 1; i < n2; ++i) {
            if (!baby_steps[i]) continue;  // Pas de clé de rotation
            
            // Créer le plaintext de la diagonale
            Message<Complex> msg_diag(log_slots, device);
            for (int k = 0; k < n; ++k) {
                int row = ((-j * n2 + k) % n + n) % n;
                int col = (k + i) % n;
                msg_diag[k] = Complex(U[row][col], 0.0);
            }
            for (int k = n; k < num_slots; ++k) {
                msg_diag[k] = Complex(0.0, 0.0);
            }
            
            auto ptxt_diag = IPlaintext::make();
            encoder.encode(msg_diag, *ptxt_diag);
            
            auto ptxt_diag_leveled = IPlaintext::make();
            eval.levelDownTo(*ptxt_diag, *ptxt_diag_leveled, eval.getLevel(*baby_steps[i]));
            
            // Multiplication
            auto ct_mul = ICiphertext::make();
            eval.mul(*baby_steps[i], *ptxt_diag_leveled, *ct_mul);
            eval.rescale(*ct_mul, *ct_mul);
            
            // Accumulation avec gestion des niveaux
            int level_gs = eval.getLevel(*ct_gs);
            int level_mul = eval.getLevel(*ct_mul);
            
            if (level_gs != level_mul) {
                if (level_gs > level_mul) {
                    auto ct_gs_leveled = ICiphertext::make();
                    eval.levelDownTo(*ct_gs, *ct_gs_leveled, level_mul);
                    ct_gs = std::move(ct_gs_leveled);
                } else {
                    auto ct_mul_leveled = ICiphertext::make();
                    eval.levelDownTo(*ct_mul, *ct_mul_leveled, level_gs);
                    ct_mul = std::move(ct_mul_leveled);
                }
            }
            
            auto ct_add = ICiphertext::make();
            eval.add(*ct_gs, *ct_mul, *ct_add);
            ct_gs = std::move(ct_add);
        }
        
        // --------------------------------------------------------
        // Rotation géante : Rot_{j·n₂}(gs_j)
        // --------------------------------------------------------
        if (j > 0) {
            int rot_amount = j * n2;
            auto it = rot_keys.find(rot_amount);
            if (it != rot_keys.end()) {
                auto ct_rot = ICiphertext::make();
                eval.rot(*ct_gs, rot_amount, *ct_rot, *(it->second));
                giant_steps[j] = std::move(ct_rot);
            } else {
                std::cerr << "  ERREUR: Clé de rotation " << rot_amount << " non trouvée!" << std::endl;
                giant_steps[j] = std::move(ct_gs);
            }
        } else {
            giant_steps[j] = std::move(ct_gs);
        }
    }
    
    // ------------------------------------------------------------
    // 5. Sommer tous les giant steps
    // ------------------------------------------------------------
    auto ct_result = ICiphertext::make();
    bool first = true;
    
    for (int j = 0; j < n1; ++j) {
        if (!giant_steps[j]) continue;
        
        if (first) {
            ct_result = std::move(giant_steps[j]);
            first = false;
        } else {
            int level_result = eval.getLevel(*ct_result);
            int level_gs = eval.getLevel(*giant_steps[j]);
            
            if (level_result != level_gs) {
                if (level_result > level_gs) {
                    auto ct_result_leveled = ICiphertext::make();
                    eval.levelDownTo(*ct_result, *ct_result_leveled, level_gs);
                    ct_result = std::move(ct_result_leveled);
                } else {
                    auto ct_gs_leveled = ICiphertext::make();
                    eval.levelDownTo(*giant_steps[j], *ct_gs_leveled, level_result);
                    giant_steps[j] = std::move(ct_gs_leveled);
                }
            }
            
            auto ct_add = ICiphertext::make();
            eval.add(*ct_result, *giant_steps[j], *ct_add);
            ct_result = std::move(ct_add);
        }
    }
    
    // ------------------------------------------------------------
    // 6. Déchiffrement
    // ------------------------------------------------------------
    auto ptxt_result = IPlaintext::make();
    encryptor.decrypt(*ct_result, sk, *ptxt_result);
    
    Message<Complex> msg_result;
    encoder.decode(*ptxt_result, msg_result);
    msg_result.to(Device::CPU);
    
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) result[i] = msg_result[i].real();
    
    return result;
}

int main() {
    std::cout << "=== Diagonal Method with BSGS (Exercice 7) ===" << std::endl;
    
    // ------------------------------------------------------------
    // 1. Génération des clés
    // ------------------------------------------------------------
    std::cout << "\n1. Génération des clés..." << std::endl;
    SKGenerator skgen(preset_id);
    auto sk = skgen.genKey();
    sk->to(device);
    
    // ------------------------------------------------------------
    // 2. Génération des clés de rotation
    // ------------------------------------------------------------
    SwKeyGenerator swkgen(preset_id);
    std::map<int, Ptr<ISwKey>> rot_keys;
    
    int n = 16;  // 4×4 pour test, mais BSGS est avantageux pour grandes dimensions
    int n1 = (int)std::sqrt(n);
    int n2 = n / n1;
    while (n1 * n2 < n) n2++;
    while (n1 * n2 > n) n1--;
    
    std::cout << "  n = " << n << " = " << n1 << " × " << n2 << std::endl;
    std::cout << "  Génération des clés de rotation..." << std::endl;
    
    // Clés pour baby steps (1..n2-1)
    for (int rot = 1; rot < n2; ++rot) {
        std::cout << "    Baby step key " << rot << "..." << std::endl;
        auto rot_key = swkgen.genRotKey(*sk, rot);
        rot_keys[rot] = std::move(rot_key);
    }
    
    // Clés pour giant steps (multiples de n2)
    for (int j = 1; j < n1; ++j) {
        int rot = j * n2;
        std::cout << "    Giant step key " << rot << "..." << std::endl;
        auto rot_key = swkgen.genRotKey(*sk, rot);
        rot_keys[rot] = std::move(rot_key);
    }
    
    // ------------------------------------------------------------
    // 3. Données de test
    // ------------------------------------------------------------
    std::cout << "\n2. Création des données de test (" << n << "×" << n << ")..." << std::endl;
    
    // Matrice identité
    std::vector<std::vector<double>> U_identity(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) U_identity[i][i] = 1.0;
    
    // Matrice aléatoire
    std::vector<std::vector<double>> U_rand(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U_rand[i][j] = (2.0 * rand() / RAND_MAX) - 1.0;
        }
    }
    
    // Vecteur v
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = i + 1.0;
    
    std::cout << "  Matrice identité créée" << std::endl;
    std::cout << "  Matrice aléatoire créée" << std::endl;
    std::cout << "  Vecteur v: [1.." << n << "]" << std::endl;
    
    // ------------------------------------------------------------
    // 4. Calcul homomorphe
    // ------------------------------------------------------------
    HomEval eval(preset_id);
    
    std::cout << "\n3. Test BSGS avec matrice identité..." << std::endl;
    std::vector<double> fhe_identity = diagonalMethodBSGS(U_identity, v, *sk, rot_keys, eval);
    
    std::cout << "\n4. Test BSGS avec matrice aléatoire..." << std::endl;
    std::vector<double> fhe_rand = diagonalMethodBSGS(U_rand, v, *sk, rot_keys, eval);
    
    // ------------------------------------------------------------
    // 5. Vérification
    // ------------------------------------------------------------
    std::cout << "\n=== Vérification Matrice Identité (BSGS) ===" << std::endl;
    
    std::vector<double> clear_identity(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            clear_identity[i] += U_identity[i][j] * v[j];
        }
    }
    
    double max_err_identity = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = std::abs(fhe_identity[i] - clear_identity[i]);
        max_err_identity = std::max(max_err_identity, err);
        if (i < 5) {
            std::cout << "  [" << i << "] Clair: " << clear_identity[i] 
                      << ", FHE: " << fhe_identity[i]
                      << ", Erreur: " << err << std::endl;
        }
    }
    
    std::cout << "\n=== Vérification Matrice Aléatoire (BSGS) ===" << std::endl;
    
    std::vector<double> clear_rand(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            clear_rand[i] += U_rand[i][j] * v[j];
        }
    }
    
    double max_err_rand = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = std::abs(fhe_rand[i] - clear_rand[i]);
        max_err_rand = std::max(max_err_rand, err);
        if (i < 5) {
            std::cout << "  [" << i << "] Clair: " << clear_rand[i] 
                      << ", FHE: " << fhe_rand[i]
                      << ", Erreur: " << err << std::endl;
        }
    }
    
    // ------------------------------------------------------------
    // 6. Statistiques
    // ------------------------------------------------------------
    std::cout << "\n=== Statistiques BSGS ===" << std::endl;
    std::cout << "Matrice Identité - Erreur max: " << max_err_identity 
              << " (log2: " << std::log2(max_err_identity) << " bits)" << std::endl;
    std::cout << "Matrice Aléatoire - Erreur max: " << max_err_rand 
              << " (log2: " << std::log2(max_err_rand) << " bits)" << std::endl;
    std::cout << "Complexité: " << (n2-1) << " baby steps + " << n1 << " giant steps = "
              << (n2-1 + n1) << " rotations" << std::endl;
    std::cout << "Au lieu de " << n << " rotations en Diagonal Method classique" << std::endl;
    std::cout << "Gain: " << (double)n / (n2-1 + n1) << "x" << std::endl;
    
    return 0;
}