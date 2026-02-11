#include "HEAAN2/HEAAN2.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace heaan;

const auto preset_id = PresetParamsId::F16Opt_Gr;
const auto device = Device::CPU;

/**
 * Rotate-and-Sum - Version 2 (SIMD)
 * 
 * @param x_vals Vecteur des x [x1, x2, ..., x_{2^k}]
 * @param coeffs Coefficients [p0, p1, ..., p_{2^k}]
 * @param sk Clé secrète
 * @param rot_key1 Clé rotation 1
 * @param rot_key2 Clé rotation 2
 * @param rot_key4 Clé rotation 4
 * @param rot_key8 Clé rotation 8
 * @param eval Évaluateur homomorphe
 */
std::vector<double> rotateAndSumSIMD(
    const std::vector<double>& x_vals,
    const std::vector<double>& coeffs,
    const ISecretKey& sk,
    const ISwKey& rot_key1,
    const ISwKey& rot_key2,
    const ISwKey& rot_key4,
    const ISwKey& rot_key8,
    HomEval& eval
) {
    int degree = coeffs.size() - 1;        // 2^k
    int num_x = x_vals.size();             // 2^k
    int k = 0;
    int temp = degree + 1;
    while (temp > 1) {
        temp >>= 1;
        k++;
    }
    
    int log_slots = sk.logDegree() - 1;
    int num_slots = 1 << log_slots;
    
    std::cout << "\n=== Rotate-and-Sum SIMD ===" << std::endl;
    std::cout << "  Degré polynôme: " << degree << ", k=" << k << std::endl;
    std::cout << "  Nombre de x: " << num_x << std::endl;
    
    // ------------------------------------------------------------
    // 1. Organisation SIMD des puissances
    // ------------------------------------------------------------
    Message<Complex> msg_powers(log_slots, device);
    
    // Pattern: pour chaque x, on met [1, x, x², ..., x^degree] dans des slots consécutifs
    int slot_idx = 0;
    for (int xi = 0; xi < num_x; ++xi) {
        double x = x_vals[xi];
        double power = 1.0;
        
        for (int i = 0; i <= degree; ++i) {
            if (slot_idx < num_slots) {
                msg_powers[slot_idx] = Complex(power, 0.0);
                power *= x;
                slot_idx++;
            }
        }
    }
    
    // Remplir le reste avec des 0
    for (int i = slot_idx; i < num_slots; ++i) {
        msg_powers[i] = Complex(0.0, 0.0);
    }
    
    // Debug : afficher les premiers slots
    std::cout << "  Organisation SIMD (premiers slots):" << std::endl;
    for (int i = 0; i < std::min(16, num_slots); i += (degree + 1)) {
        std::cout << "    x" << (i/(degree+1)) << ": ";
        for (int j = 0; j <= degree && i+j < num_slots; ++j) {
            std::cout << msg_powers[i+j].real() << " ";
        }
        std::cout << std::endl;
    }
    
    // ------------------------------------------------------------
    // 2. Encoder et chiffrer
    // ------------------------------------------------------------
    auto ptxt_powers = IPlaintext::make();
    EnDecoder encoder(preset_id);
    encoder.encode(msg_powers, *ptxt_powers);
    
    auto ct_powers = ICiphertext::make();
    EnDecryptor encryptor(preset_id);
    
    try {
        encryptor.encrypt(*ptxt_powers, sk, *ct_powers);
        std::cout << "  Encryption réussie, niveau: " 
                  << eval.getLevel(*ct_powers) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ERREUR encryption: " << e.what() << std::endl;
        throw;
    }
    
    // ------------------------------------------------------------
    // 3. Créer le plaintext des coefficients (BROADCAST)
    // ------------------------------------------------------------
    Message<Complex> msg_coeffs(log_slots, device);
    
    // Pour la version SIMD, on doit répéter les coefficients pour chaque bloc
    slot_idx = 0;
    for (int xi = 0; xi < num_x; ++xi) {
        for (int i = 0; i <= degree; ++i) {
            if (slot_idx < num_slots) {
                msg_coeffs[slot_idx] = Complex(coeffs[i], 0.0);
                slot_idx++;
            }
        }
    }
    
    // Remplir le reste avec 0
    for (int i = slot_idx; i < num_slots; ++i) {
        msg_coeffs[i] = Complex(0.0, 0.0);
    }
    
    auto ptxt_coeffs = IPlaintext::make();
    encoder.encode(msg_coeffs, *ptxt_coeffs);
    
    // ------------------------------------------------------------
    // 4. Multiplication ciphertext × plaintext
    // ------------------------------------------------------------
    auto ct_result = ICiphertext::make();
    eval.mul(*ct_powers, *ptxt_coeffs, *ct_result);
    eval.rescale(*ct_result, *ct_result);
    
    std::cout << "  Après multiplication, niveau: " 
              << eval.getLevel(*ct_result) << std::endl;
    
    // ------------------------------------------------------------
    // 5. Rotate-and-Sum loop
    // ------------------------------------------------------------
    const ISwKey* rot_keys[4] = {&rot_key1, &rot_key2, &rot_key4, &rot_key8};
    
    for (int i = 0; i < k; ++i) {
        int rot_amount = 1 << i;
        std::cout << "    Rotation de " << rot_amount << " positions" << std::endl;
        
        auto ct_rot = ICiphertext::make();
        eval.rot(*ct_result, rot_amount, *ct_rot, *rot_keys[i]);
        
        auto ct_sum = ICiphertext::make();
        eval.add(*ct_result, *ct_rot, *ct_sum);
        *ct_result = *ct_sum;  // Copie du contenu
        
        std::cout << "      Niveau après addition: " 
                  << eval.getLevel(*ct_result) << std::endl;
    }
    
    // ------------------------------------------------------------
    // 6. Déchiffrement et extraction des résultats
    // ------------------------------------------------------------
    auto ptxt_result = IPlaintext::make();
    encryptor.decrypt(*ct_result, sk, *ptxt_result);
    
    Message<Complex> msg_result;
    encoder.decode(*ptxt_result, msg_result);
    msg_result.to(Device::CPU);
    
    // Extraire les résultats (slot 0, slot degree+1, slot 2*(degree+1), ...)
    std::vector<double> results;
    for (int i = 0; i < num_x; ++i) {
        int slot = i * (degree + 1);
        if (slot < num_slots) {
            results.push_back(msg_result[slot].real());
        }
    }
    
    // Debug : afficher les résultats
    std::cout << "\n  Résultats extraits (slot 0 de chaque bloc):" << std::endl;
    for (size_t i = 0; i < results.size() && i < 4; ++i) {
        std::cout << "    P(x" << i << ") = " << results[i] << std::endl;
    }
    
    return results;
}

int main() {
    std::cout << "=== Rotate-and-Sum SIMD (Exercice 4 - Version 2) ===" << std::endl;
    
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
    
    std::cout << "  Génération clé rotation 1..." << std::endl;
    auto rot_key1 = swkgen.genRotKey(*sk, 1);
    
    std::cout << "  Génération clé rotation 2..." << std::endl;
    auto rot_key2 = swkgen.genRotKey(*sk, 2);
    
    std::cout << "  Génération clé rotation 4..." << std::endl;
    auto rot_key4 = swkgen.genRotKey(*sk, 4);
    
    std::cout << "  Génération clé rotation 8..." << std::endl;
    auto rot_key8 = swkgen.genRotKey(*sk, 8);
    
    // ------------------------------------------------------------
    // 3. Données de test
    // ------------------------------------------------------------
    std::cout << "\n2. Création des données de test..." << std::endl;
    
    // Polynôme P(x) = 1 + x + x² + x³ (degré 3)
    std::vector<double> coeffs = {1.0, 1.0, 1.0, 1.0};
    
    // 4 valeurs de x
    std::vector<double> x_vals = {1.0, 2.0, 3.0, 4.0};
    
    std::cout << "  Polynôme: P(x) = ";
    for (size_t i = 0; i < coeffs.size(); ++i) {
        if (i > 0) std::cout << " + ";
        std::cout << coeffs[i] << "x^" << i;
    }
    std::cout << std::endl;
    
    std::cout << "  x: ";
    for (double x : x_vals) std::cout << x << " ";
    std::cout << std::endl;
    
    // ------------------------------------------------------------
    // 4. Calcul homomorphe
    // ------------------------------------------------------------
    HomEval eval(preset_id);
    
    std::vector<double> fhe_results = rotateAndSumSIMD(
        x_vals, coeffs, *sk, 
        *rot_key1, *rot_key2, *rot_key4, *rot_key8, 
        eval
    );
    
    // ------------------------------------------------------------
    // 5. Calcul en clair pour vérification
    // ------------------------------------------------------------
    std::cout << "\n3. Vérification avec calcul en clair:" << std::endl;
    
    double max_err = 0.0;
    for (size_t i = 0; i < x_vals.size() && i < fhe_results.size(); ++i) {
        double x = x_vals[i];
        
        // Calcul en clair de P(x)
        double clear_result = 0.0;
        for (size_t j = 0; j < coeffs.size(); ++j) {
            clear_result += coeffs[j] * std::pow(x, j);
        }
        
        double err = std::abs(fhe_results[i] - clear_result);
        max_err = std::max(max_err, err);
        
        std::cout << "  x=" << x 
                  << " | Clair=" << clear_result 
                  << " | FHE=" << fhe_results[i]
                  << " | Erreur=" << err << std::endl;
    }
    
    std::cout << "\n=== Statistiques ===" << std::endl;
    std::cout << "Erreur maximale: " << max_err << std::endl;
    std::cout << "Erreur (log2): " << std::log2(max_err) << " bits" << std::endl;
    
    return 0;
}