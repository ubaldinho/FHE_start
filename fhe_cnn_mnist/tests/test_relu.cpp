#include "fhe_cnn/relu.hpp"
#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <chrono>
#include <random>

using namespace heaan;
using namespace fhe_cnn;

int main() {
    std::cout << "\n🧪 Test ReLU Approximation" << std::endl;
    std::cout << "==========================" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // ------------------------------------------------------------
    // 1. Initialisation HEAAN2
    // ------------------------------------------------------------
    std::cout << "\n1. Initialisation HEAAN2..." << std::endl;
    
    auto preset_id = PresetParamsId::F16Opt_Gr;
    
    SKGenerator skgen(preset_id);
    auto sk = skgen.genKey();
    sk->to(Device::CPU);
    
    SwKeyGenerator swkgen(preset_id);
    auto relin_key = swkgen.genRelinKey(*sk);
    
    HomEval eval(preset_id);
    EnDecoder encoder(preset_id);
    EnDecryptor encryptor(preset_id);
    
    // ------------------------------------------------------------
    // 2. Création des données de test
    // ------------------------------------------------------------
    std::cout << "\n2. Création des données de test..." << std::endl;
    
    std::vector<double> input(16);
    for (int i = 0; i < 16; ++i) {
        input[i] = -0.8 + i * 0.1;  // De -0.8 à 0.7
    }
    
    double scale_factor = 1.0;  // Déjà dans [-1,1]
    
    // ------------------------------------------------------------
    // 3. Chiffrement
    // ------------------------------------------------------------
    std::cout << "\n3. Chiffrement..." << std::endl;
    
    auto ct_input = encrypt_image(input, *sk, encoder, encryptor);
    
    // ------------------------------------------------------------
    // 4. ReLU homomorphe (degré 3,5,7)
    // ------------------------------------------------------------
    std::cout << "\n4. Exécution ReLU degré 3..." << std::endl;
    auto ct_relu3 = homomorphic_relu(*ct_input, 3, scale_factor, eval, *relin_key);
    
    std::cout << "\n5. Exécution ReLU degré 5..." << std::endl;
    auto ct_relu5 = homomorphic_relu(*ct_input, 5, scale_factor, eval, *relin_key);
    
    // ------------------------------------------------------------
    // 5. Déchiffrement
    // ------------------------------------------------------------
    std::cout << "\n6. Déchiffrement..." << std::endl;
    
    auto output3 = decrypt_result(*ct_relu3, *sk, encoder, encryptor, input.size());
    auto output5 = decrypt_result(*ct_relu5, *sk, encoder, encryptor, input.size());
    
    // ------------------------------------------------------------
    // 6. Vérification
    // ------------------------------------------------------------
    std::cout << "\n7. Vérification..." << std::endl;
    
    std::vector<double> relu_true(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        relu_true[i] = std::max(0.0, input[i]);
    }
    
    std::cout << "\n=== Résultats (degré 3) ===" << std::endl;
    double max_err3 = 0.0;
    for (size_t i = 0; i < std::min((size_t)5, input.size()); ++i) {
        double err = std::abs(output3[i] - relu_true[i]);
        max_err3 = std::max(max_err3, err);
        std::cout << "  x=" << input[i] 
                  << ", ReLU vrai=" << relu_true[i]
                  << ", ReLU3=" << output3[i]
                  << ", err=" << err << std::endl;
    }
    
    std::cout << "\n=== Résultats (degré 5) ===" << std::endl;
    double max_err5 = 0.0;
    for (size_t i = 0; i < std::min((size_t)5, input.size()); ++i) {
        double err = std::abs(output5[i] - relu_true[i]);
        max_err5 = std::max(max_err5, err);
        std::cout << "  x=" << input[i] 
                  << ", ReLU vrai=" << relu_true[i]
                  << ", ReLU5=" << output5[i]
                  << ", err=" << err << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n=== Statistiques ===" << std::endl;
    std::cout << "  Erreur max degré 3: " << max_err3 << std::endl;
    std::cout << "  Erreur max degré 5: " << max_err5 << std::endl;
    std::cout << "  Temps: " << duration.count() << " ms" << std::endl;
    
    return 0;
}