#include "HEAAN2/HEAAN2.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace heaan;

const auto preset_id = PresetParamsId::F16Opt_Gr;
const auto device = Device::CPU;

/**
 * Rotate-and-Sum - Version 1 (scalaire)
 * 
 * @param x Valeur scalaire à évaluer
 * @param coeffs Coefficients [p0, p1, ..., p_{2^k}]
 * @param sk Clé secrète
 * @param rot_key1 Clé pour rotation 1
 * @param rot_key2 Clé pour rotation 2
 * @param rot_key4 Clé pour rotation 4
 * @param rot_key8 Clé pour rotation 8 (si nécessaire)
 * @param eval Évaluateur homomorphe
 */
double rotateAndSumScalar(
    double x,
    const std::vector<double>& coeffs,
    const ISecretKey& sk,
    const ISwKey& rot_key1,
    const ISwKey& rot_key2,
    const ISwKey& rot_key4,
    const ISwKey& rot_key8,
    HomEval& eval
) {
    int degree = coeffs.size() - 1;
    int k = 0;
    int temp = degree + 1;
    while (temp > 1) {
        temp >>= 1;
        k++;
    }
    
    int log_slots = sk.logDegree() - 1;
    int num_slots = 1 << log_slots;
    
    std::cout << "\n=== Rotate-and-Sum (scalaire) ===" << std::endl;
    std::cout << "  Degré: " << degree << ", k=" << k << std::endl;
    
    // ------------------------------------------------------------
    // 1. Créer le message avec [1, x, x², ..., x^{2^k}, 0, ...]
    // ------------------------------------------------------------
    Message<Complex> msg_powers(log_slots, device);
    
    double power = 1.0;
    for (int i = 0; i <= degree; ++i) {
        msg_powers[i] = Complex(power, 0.0);
        power *= x;
    }
    
    // Afficher les premières puissances pour debug
    std::cout << "  Puissances: 1";
    for (int i = 1; i <= std::min(4, degree); ++i) {
        std::cout << ", " << msg_powers[i].real();
    }
    std::cout << std::endl;
    
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
        std::cout << "  Encryption réussie" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ERREUR encryption: " << e.what() << std::endl;
        throw;
    }
    
    // Vérifier que ct_powers est valide via getLevel
    try {
        int level = eval.getLevel(*ct_powers);
        std::cout << "  Message des puissances chiffré, niveau: " << level << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ERREUR: ct_powers est vide ou invalide!" << std::endl;
        throw;
    }
    
    // ------------------------------------------------------------
    // 3. Déchiffrer pour vérifier le contenu (debug)
    // ------------------------------------------------------------
    auto ptxt_check = IPlaintext::make();
    encryptor.decrypt(*ct_powers, sk, *ptxt_check);
    Message<Complex> msg_check;
    encoder.decode(*ptxt_check, msg_check);
    msg_check.to(Device::CPU);
    std::cout << "  Vérification - slot0: " << msg_check[0].real() 
              << ", slot1: " << msg_check[1].real() << std::endl;
    
    // ------------------------------------------------------------
    // 4. Créer le plaintext des coefficients
    // ------------------------------------------------------------
    Message<Complex> msg_coeffs(log_slots, device);
    for (int i = 0; i < num_slots; ++i) {
        if (i <= degree) {
            msg_coeffs[i] = Complex(coeffs[i], 0.0);
        } else {
            msg_coeffs[i] = Complex(0.0, 0.0);
        }
    }
    
    auto ptxt_coeffs = IPlaintext::make();
    encoder.encode(msg_coeffs, *ptxt_coeffs);
    
    // ------------------------------------------------------------
    // 5. Multiplication ciphertext × plaintext
    // ------------------------------------------------------------
    auto ct_result = ICiphertext::make();
    
    try {
        eval.mul(*ct_powers, *ptxt_coeffs, *ct_result);
        std::cout << "  Multiplication réussie" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ERREUR multiplication: " << e.what() << std::endl;
        throw;
    }
    
    try {
        eval.rescale(*ct_result, *ct_result);
        std::cout << "  Rescale réussi" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ERREUR rescale: " << e.what() << std::endl;
        throw;
    }
    
    int level_after_mul = eval.getLevel(*ct_result);
    std::cout << "  Après multiplication, niveau: " << level_after_mul << std::endl;
    
    // ------------------------------------------------------------
    // 6. Rotate-and-Sum loop
    // ------------------------------------------------------------
    const ISwKey* rot_keys[4] = {&rot_key1, &rot_key2, &rot_key4, &rot_key8};
    
    for (int i = 0; i < k; ++i) {
        int rot_amount = 1 << i;
        std::cout << "    Rotation de " << rot_amount << " positions" << std::endl;
        
        auto ct_rot = ICiphertext::make();
        
        try {
            eval.rot(*ct_result, rot_amount, *ct_rot, *rot_keys[i]);
            std::cout << "      Rotation réussie" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "      ERREUR rotation: " << e.what() << std::endl;
            throw;
        }
        
        // Vérifier que ct_rot est valide
        int level_rot = eval.getLevel(*ct_rot);
        std::cout << "      Niveau après rotation: " << level_rot << std::endl;
        
        auto ct_sum = ICiphertext::make();
        eval.add(*ct_result, *ct_rot, *ct_sum);
        *ct_result = *ct_sum;  // Copie du contenu
        
        int level_sum = eval.getLevel(*ct_result);
        std::cout << "      Niveau après addition: " << level_sum << std::endl;
    }
    
    // ------------------------------------------------------------
    // 7. Déchiffrement
    // ------------------------------------------------------------
    auto ptxt_result = IPlaintext::make();
    encryptor.decrypt(*ct_result, sk, *ptxt_result);
    
    Message<Complex> msg_result;
    encoder.decode(*ptxt_result, msg_result);
    msg_result.to(Device::CPU);
    
    double result = msg_result[0].real();
    std::cout << "  Résultat slot 0: " << result << std::endl;
    
    return result;
}

int main() {
    std::cout << "=== Rotate-and-Sum (Exercice 4) ===" << std::endl;
    
    // 1. Génération des clés
    std::cout << "\n1. Génération des clés..." << std::endl;
    SKGenerator skgen(preset_id);
    auto sk = skgen.genKey();
    sk->to(device);
    
    // 2. Génération des clés de rotation pour 1,2,4,8
    SwKeyGenerator swkgen(preset_id);
    
    std::cout << "  Génération clé rotation 1..." << std::endl;
    auto rot_key1 = swkgen.genRotKey(*sk, 1);
    
    std::cout << "  Génération clé rotation 2..." << std::endl;
    auto rot_key2 = swkgen.genRotKey(*sk, 2);
    
    std::cout << "  Génération clé rotation 4..." << std::endl;
    auto rot_key4 = swkgen.genRotKey(*sk, 4);
    
    std::cout << "  Génération clé rotation 8..." << std::endl;
    auto rot_key8 = swkgen.genRotKey(*sk, 8);
    
    // 3. Test avec un polynôme simple
    std::cout << "\n2. Test du polynôme P(x) = 1 + x + x² + x³ + x⁴" << std::endl;
    std::vector<double> coeffs = {1.0, 1.0, 1.0, 1.0, 1.0};  // degré 4
    double x = 2.0;
    
    HomEval eval(preset_id);
    
    double result_fhe = rotateAndSumScalar(x, coeffs, *sk, 
                                          *rot_key1, *rot_key2, *rot_key4, *rot_key8, 
                                          eval);
    
    // Calcul en clair pour vérification
    double result_clear = 0.0;
    for (size_t i = 0; i < coeffs.size(); ++i) {
        result_clear += coeffs[i] * std::pow(x, i);
    }
    
    std::cout << "\n=== Vérification ===" << std::endl;
    std::cout << "P(" << x << ") clair = " << result_clear << std::endl;
    std::cout << "P(" << x << ") FHE   = " << result_fhe << std::endl;
    std::cout << "Erreur = " << std::abs(result_fhe - result_clear) << std::endl;
    
    return 0;
}
