#include "HEAAN2/HEAAN2.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <map>

using namespace heaan;

const auto preset_id = PresetParamsId::F16Opt_Gr;
const auto device = Device::CPU;

std::vector<double> rowMethod(
    const std::vector<std::vector<double>>& U,
    const std::vector<double>& v,
    const ISecretKey& sk,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval
) {
    int n = U.size();
    int log_slots = sk.logDegree() - 1;
    int num_slots = 1 << log_slots;
    
    std::cout << "\n=== Row Method ===" << std::endl;
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
    // 2. Plaintext pour masquer (1,0,0,...)
    // ------------------------------------------------------------
    Message<Complex> msg_mask(log_slots, device);
    msg_mask[0] = Complex(1.0, 0.0);
    for (int i = 1; i < num_slots; ++i) msg_mask[i] = Complex(0.0, 0.0);
    
    auto ptxt_mask = IPlaintext::make();
    encoder.encode(msg_mask, *ptxt_mask);
    
    // ------------------------------------------------------------
    // 3. Pour chaque ligne
    // ------------------------------------------------------------
    auto ct_result = ICiphertext::make();  // Ciphertext vide au début
    bool first = true;
    
    for (int i = 0; i < n; ++i) {
        if (i % 10 == 0) std::cout << "  Traitement ligne " << i << "/" << n << std::endl;
        
        // --------------------------------------------------------
        // 3a. Plaintext de la ligne i
        // --------------------------------------------------------
        Message<Complex> msg_row(log_slots, device);
        for (int j = 0; j < n; ++j) msg_row[j] = Complex(U[i][j], 0.0);
        for (int j = n; j < num_slots; ++j) msg_row[j] = Complex(0.0, 0.0);
        
        auto ptxt_row = IPlaintext::make();
        encoder.encode(msg_row, *ptxt_row);
        
        auto ptxt_row_leveled = IPlaintext::make();
        eval.levelDownTo(*ptxt_row, *ptxt_row_leveled, level_v);
        
        // --------------------------------------------------------
        // 3b. z_i = ligne_i ⊙ ct_v
        // --------------------------------------------------------
        auto ct_sum = ICiphertext::make();
        eval.mul(*ct_v, *ptxt_row_leveled, *ct_sum);
        
        // --------------------------------------------------------
        // 3c. Sommer tous les slots (rotate-and-sum)
        // --------------------------------------------------------
        for (int shift = 1; shift < n; shift <<= 1) {
            auto it = rot_keys.find(shift);
            if (it != rot_keys.end()) {
                auto ct_rot = ICiphertext::make();
                eval.rot(*ct_sum, shift, *ct_rot, *(it->second));
                
                auto ct_new = ICiphertext::make();
                eval.add(*ct_sum, *ct_rot, *ct_new);
                *ct_sum = *ct_new;
            }
        }
        
        // --------------------------------------------------------
        // 3d. Rescale
        // --------------------------------------------------------
        eval.rescale(*ct_sum, *ct_sum);
        
        // --------------------------------------------------------
        // 3e. Extraire slot 0
        // --------------------------------------------------------
        auto ct_extract = ICiphertext::make();
        
        auto ptxt_mask_current = IPlaintext::make();
        eval.levelDownTo(*ptxt_mask, *ptxt_mask_current, eval.getLevel(*ct_sum));
        
        eval.mul(*ct_sum, *ptxt_mask_current, *ct_extract);
        eval.rescale(*ct_extract, *ct_extract);
        
        // --------------------------------------------------------
        // 3f. Rotation à la position i
        // --------------------------------------------------------
        if (i > 0) {
            auto it = rot_keys.find(i);
            if (it != rot_keys.end()) {
                auto ct_rotated = ICiphertext::make();
                eval.rot(*ct_extract, i, *ct_rotated, *(it->second));
                ct_extract = std::move(ct_rotated);
            }
        }
        
        // --------------------------------------------------------
        // 3g. Accumuler
        // --------------------------------------------------------
        if (first) {
            ct_result = std::move(ct_extract);  // Premier résultat
            first = false;
            std::cout << "    Première ligne ajoutée, niveau: " << eval.getLevel(*ct_result) << std::endl;
        } else {
            std::cout << "    Accumulation - ct_result niveau: " << eval.getLevel(*ct_result) 
                      << ", ct_extract niveau: " << eval.getLevel(*ct_extract) << std::endl;
            
            auto ct_add = ICiphertext::make();
            eval.add(*ct_result, *ct_extract, *ct_add);
            ct_result = std::move(ct_add);
            std::cout << "    Accumulation OK, niveau: " << eval.getLevel(*ct_result) << std::endl;
        }
    }
    
    // ------------------------------------------------------------
    // 4. Déchiffrement
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
    std::cout << "=== Row Method (Exercice 5) ===" << std::endl;
    
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
    
    for (int rot = 1; rot < n; rot <<= 1) {
        std::cout << "  Génération clé rotation " << rot << "..." << std::endl;
        auto rot_key = swkgen.genRotKey(*sk, rot);
        rot_keys[rot] = std::move(rot_key);
    }
    
    for (int rot = 1; rot < n; ++rot) {
        if (rot_keys.find(rot) == rot_keys.end()) {
            std::cout << "  Génération clé rotation " << rot << "..." << std::endl;
            auto rot_key = swkgen.genRotKey(*sk, rot);
            rot_keys[rot] = std::move(rot_key);
        }
    }
    
    // ------------------------------------------------------------
    // 3. Données de test
    // ------------------------------------------------------------
    std::cout << "\n2. Création des données de test (" << n << "×" << n << ")..." << std::endl;
    
    std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) U[i][i] = 1.0;
    
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = i + 1.0;
    
    std::cout << "  Matrice: identité" << std::endl;
    std::cout << "  Vecteur v: ";
    for (double x : v) std::cout << x << " ";
    std::cout << std::endl;
    
    // ------------------------------------------------------------
    // 4. Calcul homomorphe
    // ------------------------------------------------------------
    HomEval eval(preset_id);
    
    std::vector<double> fhe_result = rowMethod(U, v, *sk, rot_keys, eval);
    
    // ------------------------------------------------------------
    // 5. Vérification
    // ------------------------------------------------------------
    std::cout << "\n3. Vérification:" << std::endl;
    
    std::vector<double> clear_result(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            clear_result[i] += U[i][j] * v[j];
        }
    }
    
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = std::abs(fhe_result[i] - clear_result[i]);
        max_err = std::max(max_err, err);
        
        std::cout << "  [" << i << "] Clair: " << clear_result[i] 
                  << ", FHE: " << fhe_result[i]
                  << ", Erreur: " << err << std::endl;
    }
    
    std::cout << "\n=== Statistiques ===" << std::endl;
    std::cout << "Erreur maximale: " << max_err << std::endl;
    std::cout << "Erreur (log2): " << std::log2(max_err) << " bits" << std::endl;
    
    return 0;
}