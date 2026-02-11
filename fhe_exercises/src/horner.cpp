#include "HEAAN2/HEAAN2.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace heaan;

const auto preset_id = PresetParamsId::F16Opt_Gr;
const auto device = Device::CPU;

Ptr<ICiphertext> homomorphicHorner(
    const ICiphertext& ct_x,
    const std::vector<double>& coeffs,
    const ISecretKey& sk,
    const ISwKey& relin_key,
    HomEval& eval
) {
    int d = coeffs.size() - 1;
    
    std::cout << "  Degré du polynôme: " << d << std::endl;
    
    // 1. Initialiser ct' avec p_d (chiffré)
    Message<Complex> msg_pd(sk.logDegree() - 1, device);
    for (int i = 0; i < (1 << msg_pd.logSlots()); ++i) {
        msg_pd[i] = Complex(coeffs[d], 0.0);
    }
    
    auto ptxt_pd = IPlaintext::make();
    EnDecoder encoder(preset_id);
    encoder.encode(msg_pd, *ptxt_pd);
    
    // IMPORTANT : Créer et remplir ct_prime
    auto ct_prime = ICiphertext::make();
    EnDecryptor encryptor(preset_id);
    encryptor.encrypt(*ptxt_pd, sk, *ct_prime);
    
    // Vérifier que ct_prime n'est pas vide
    try {
        int level = eval.getLevel(*ct_prime);
        std::cout << "  Niveau initial ct': " << level << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Erreur: ct' est vide après encryption!" << std::endl;
        throw;
    }
    
    // 2. Mettre ct' au même niveau que ct_x
    int target_level = eval.getLevel(ct_x);
    std::cout << "  Niveau ct_x: " << target_level << std::endl;
    
    auto ct_prime_leveled = ICiphertext::make();
    eval.levelDownTo(*ct_prime, *ct_prime_leveled, target_level);
    *ct_prime = *ct_prime_leveled;
    
    std::cout << "  Niveau ct' après levelDown: " << eval.getLevel(*ct_prime) << std::endl;
    
    // 3. Boucle de Horner
    for (int i = d - 1; i >= 0; --i) {
        std::cout << "\n  --- Itération i=" << i << " (p_" << i << " = " << coeffs[i] << ") ---" << std::endl;
        
        // Créer ct_x_current au niveau actuel de ct_prime
        auto ct_x_current = ICiphertext::make();
        eval.levelDownTo(ct_x, *ct_x_current, eval.getLevel(*ct_prime));
        
        // Multiplication
        auto temp = ICiphertext::make();
        eval.tensor(*ct_x_current, *ct_prime, *temp);
        eval.relin(*temp, relin_key);
        eval.rescale(*temp, *temp);
        
        // Addition avec constante
        auto ct_temp = ICiphertext::make();
        eval.add(*temp, Complex(coeffs[i], 0.0), *ct_temp);
        *ct_prime = *ct_temp;
        
        std::cout << "    Niveau ct' après itération: " << eval.getLevel(*ct_prime) << std::endl;
    }
    
    return ct_prime;
}

int main() {
    std::cout << "=== Horner homomorphe (version finale) ===" << std::endl;
    
    // 1. Clés
    std::cout << "1. Génération des clés..." << std::endl;
    SKGenerator skgen(preset_id);
    auto sk = skgen.genKey();
    sk->to(device);
    
    SwKeyGenerator swkgen(preset_id);
    auto relin_key = swkgen.genRelinKey(*sk);
    
    // 2. Données
    int log_slots = sk->logDegree() - 1;
    int num_slots = 1 << log_slots;
    std::cout << "2. Création des données (" << num_slots << " slots)..." << std::endl;
    
    Message<Complex> msg_x(log_slots, device);
    for (int i = 0; i < num_slots; ++i) {
        double val = (2.0 * rand() / RAND_MAX) - 1.0;
        msg_x[i] = Complex(val, 0.0);
    }
    
    // 3. Chiffrement
    auto ptxt_x = IPlaintext::make();
    EnDecoder encoder(preset_id);
    encoder.encode(msg_x, *ptxt_x);
    
    auto ct_x = ICiphertext::make();
    EnDecryptor encryptor(preset_id);
    encryptor.encrypt(*ptxt_x, *sk, *ct_x);
    
    // 4. Polynôme test
    std::vector<double> coeffs = {-5.0, 1.0, -3.0, 2.0}; // degré 3
    int d = coeffs.size() - 1;
    
    // 5. Évaluation homomorphe
    std::cout << "3. Évaluation homomorphe..." << std::endl;
    HomEval eval(preset_id);
    
    auto ct_result = homomorphicHorner(*ct_x, coeffs, *sk, *relin_key, eval);
    
    // 6. Déchiffrement
    std::cout << "\n4. Déchiffrement..." << std::endl;
    auto ptxt_result = IPlaintext::make();
    encryptor.decrypt(*ct_result, *sk, *ptxt_result);
    
    Message<Complex> msg_result;
    encoder.decode(*ptxt_result, msg_result);
    msg_result.to(Device::CPU);
    
    // 7. Vérification
    std::cout << "\n=== Vérification (5 premiers slots) ===" << std::endl;
    
    double max_err = 0.0;
    for (int i = 0; i < std::min(5, num_slots); ++i) {
        double x = msg_x[i].real();
        
        // Calcul en clair
        double y_clear = 0.0;
        for (int j = d; j >= 0; --j) {
            y_clear = y_clear * x + coeffs[j];
        }
        
        double y_fhe = msg_result[i].real();
        double err = std::abs(y_fhe - y_clear);
        if (err > max_err) max_err = err;
        
        std::cout << "Slot " << i << ": x=" << x 
                  << ", Clair=" << y_clear
                  << ", FHE=" << y_fhe
                  << ", Erreur=" << err << std::endl;
    }
    
    std::cout << "\n=== Statistiques finales ===" << std::endl;
    std::cout << "Erreur maximale: " << max_err << std::endl;
    std::cout << "Erreur log2: " << std::log2(max_err) << " bits" << std::endl;
    std::cout << "Niveau final: " << eval.getLevel(*ct_result) << std::endl;
    
    return 0;
}