#include "HEAAN2/HEAAN2.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <map>

using namespace heaan;

const auto preset_id = PresetParamsId::F16Opt_Gr;
const auto device = Device::CPU;

std::vector<double> diagonalMethod(
    const std::vector<std::vector<double>>& U,
    const std::vector<double>& v,
    const ISecretKey& sk,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval
) {
    int n = U.size();
    int log_slots = sk.logDegree() - 1;
    int num_slots = 1 << log_slots;
    
    std::cout << "\n=== Diagonal Method ===" << std::endl;
    std::cout << "  Dimension: " << n << "×" << n << std::endl;
    
    // ------------------------------------------------------------
    // 1. Chiffrer le vecteur v
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
    // 2. Pour chaque diagonale i = 0..n-1
    // ------------------------------------------------------------
    auto ct_result = ICiphertext::make();
    bool first = true;
    
    for (int i = 0; i < n; ++i) {
        if (i % 10 == 0) std::cout << "  Traitement diagonale " << i << "/" << n << std::endl;
        
        if (i == 0) {
            // --------------------------------------------------------
            // Cas i=0 : pas de rotation, diagonale principale
            // --------------------------------------------------------
            Message<Complex> msg_diag(log_slots, device);
            for (int j = 0; j < n; ++j) {
                msg_diag[j] = Complex(U[j][j], 0.0);  // Ligne j, colonne j
            }
            for (int j = n; j < num_slots; ++j) {
                msg_diag[j] = Complex(0.0, 0.0);
            }
            
            auto ptxt_diag = IPlaintext::make();
            encoder.encode(msg_diag, *ptxt_diag);
            
            auto ptxt_diag_leveled = IPlaintext::make();
            eval.levelDownTo(*ptxt_diag, *ptxt_diag_leveled, eval.getLevel(*ct_v));
            
            auto ct_mul = ICiphertext::make();
            eval.mul(*ct_v, *ptxt_diag_leveled, *ct_mul);
            eval.rescale(*ct_mul, *ct_mul);
            
            ct_result = std::move(ct_mul);
            first = false;
            std::cout << "    Diagonale 0 traitée, niveau: " << eval.getLevel(*ct_result) << std::endl;
            
        } else {
            // --------------------------------------------------------
            // Cas i>0 : rotation nécessaire
            // --------------------------------------------------------
            auto it = rot_keys.find(i);
            if (it == rot_keys.end()) {
                std::cerr << "  ERREUR: Clé de rotation " << i << " non trouvée!" << std::endl;
                continue;
            }
            
            // Rotation du vecteur v
            auto ct_rot = ICiphertext::make();
            eval.rot(*ct_v, i, *ct_rot, *(it->second));
            
            // --------------------------------------------------------
            // Créer le plaintext de la diagonale i
            // CORRECTION FINALE : ligne = j, colonne = (j + i) % n
            // --------------------------------------------------------
            Message<Complex> msg_diag(log_slots, device);
            for (int j = 0; j < n; ++j) {
                int col = (j + i) % n;  // Colonne = (j + i) mod n
                msg_diag[j] = Complex(U[j][col], 0.0);  // Ligne j, colonne (j+i)%n
            }
            for (int j = n; j < num_slots; ++j) {
                msg_diag[j] = Complex(0.0, 0.0);
            }
            
            auto ptxt_diag = IPlaintext::make();
            encoder.encode(msg_diag, *ptxt_diag);
            
            auto ptxt_diag_leveled = IPlaintext::make();
            eval.levelDownTo(*ptxt_diag, *ptxt_diag_leveled, eval.getLevel(*ct_rot));
            
            // Multiplication : diagonale ⊙ Rot_i(v)
            auto ct_mul = ICiphertext::make();
            eval.mul(*ct_rot, *ptxt_diag_leveled, *ct_mul);
            eval.rescale(*ct_mul, *ct_mul);
            
            // --------------------------------------------------------
            // Accumulation
            // --------------------------------------------------------
            if (first) {
                ct_result = std::move(ct_mul);
                first = false;
            } else {
                int level_result = eval.getLevel(*ct_result);
                int level_mul = eval.getLevel(*ct_mul);
                
                if (level_result != level_mul) {
                    if (level_result > level_mul) {
                        auto ct_result_leveled = ICiphertext::make();
                        eval.levelDownTo(*ct_result, *ct_result_leveled, level_mul);
                        ct_result = std::move(ct_result_leveled);
                    } else {
                        auto ct_mul_leveled = ICiphertext::make();
                        eval.levelDownTo(*ct_mul, *ct_mul_leveled, level_result);
                        ct_mul = std::move(ct_mul_leveled);
                    }
                }
                
                auto ct_add = ICiphertext::make();
                eval.add(*ct_result, *ct_mul, *ct_add);
                ct_result = std::move(ct_add);
            }
            
            std::cout << "    Diagonale " << i << " traitée, niveau: " 
                      << eval.getLevel(*ct_result) << std::endl;
        }
    }
    
    // ------------------------------------------------------------
    // 3. Déchiffrement
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
    std::cout << "=== Diagonal Method (Exercice 6) ===" << std::endl;
    
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
    
    int n = 4;
    
    for (int rot = 1; rot < n; ++rot) {
        std::cout << "  Génération clé rotation " << rot << "..." << std::endl;
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
    std::cout << "  Vecteur v: ";
    for (double x : v) std::cout << x << " ";
    std::cout << std::endl;
    
    // ------------------------------------------------------------
    // 4. Calcul homomorphe
    // ------------------------------------------------------------
    HomEval eval(preset_id);
    
    std::cout << "\n3. Test avec matrice identité..." << std::endl;
    std::vector<double> fhe_identity = diagonalMethod(U_identity, v, *sk, rot_keys, eval);
    
    std::cout << "\n4. Test avec matrice aléatoire..." << std::endl;
    std::vector<double> fhe_rand = diagonalMethod(U_rand, v, *sk, rot_keys, eval);
    
    // ------------------------------------------------------------
    // 5. Vérification
    // ------------------------------------------------------------
    std::cout << "\n=== Vérification Matrice Identité ===" << std::endl;
    
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
        std::cout << "  [" << i << "] Clair: " << clear_identity[i] 
                  << ", FHE: " << fhe_identity[i]
                  << ", Erreur: " << err << std::endl;
    }
    
    std::cout << "\n=== Vérification Matrice Aléatoire ===" << std::endl;
    
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
        std::cout << "  [" << i << "] Clair: " << clear_rand[i] 
                  << ", FHE: " << fhe_rand[i]
                  << ", Erreur: " << err << std::endl;
    }
    
    std::cout << "\n=== Statistiques ===" << std::endl;
    std::cout << "Matrice Identité - Erreur max: " << max_err_identity 
              << " (log2: " << std::log2(max_err_identity) << " bits)" << std::endl;
    std::cout << "Matrice Aléatoire - Erreur max: " << max_err_rand 
              << " (log2: " << std::log2(max_err_rand) << " bits)" << std::endl;
    
    return 0;
}