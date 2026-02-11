#include "fhe_cnn/fc.hpp"
#include "fhe_cnn/conv2d.hpp"
#include "fhe_cnn/pooling.hpp"
#include "fhe_cnn/relu.hpp"
#include "fhe_cnn/bootstrapping.hpp"
#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <chrono>
#include <memory>
#include <iomanip>

using namespace heaan;
using namespace fhe_cnn;

int main() {
    std::cout << "\n🚀 FHE CNN MNIST - Projet 5CS09" << std::endl;
    std::cout << "=================================" << std::endl;
    
    auto program_start = std::chrono::high_resolution_clock::now();
    
    // ------------------------------------------------------------
    // 1. Initialisation HEAAN2
    // ------------------------------------------------------------
    std::cout << "\n1. Initialisation HEAAN2..." << std::endl;
    
    auto preset_id = PresetParamsId::F16Opt_Gr;
    
    // Génération de la clé secrète
    SKGenerator skgen(preset_id);
    auto sk = skgen.genKey();
    sk->to(Device::CPU);
    
    // Génération de la clé de relinéarisation
    SwKeyGenerator swkgen(preset_id);
    auto relin_key = swkgen.genRelinKey(*sk);
    
    // Évaluateur homomorphe
    HomEval eval(preset_id);
    EnDecoder encoder(preset_id);
    EnDecryptor encryptor(preset_id);
    
    int log_slots = sk->logDegree() - 1;
    int num_slots = 1 << log_slots;
    
    std::cout << "    logDegree: " << sk->logDegree() << std::endl;
    std::cout << "    logSlots: " << log_slots << std::endl;
    std::cout << "    numSlots: " << num_slots << std::endl;
    
    // ------------------------------------------------------------
    // 2. Chargement des données MNIST et poids
    // ------------------------------------------------------------
    std::cout << "\n2. Chargement des données..." << std::endl;
    
    try {
        auto images = load_mnist_images("data/mnist/t10k-images-idx3-ubyte");
        auto labels = load_mnist_labels("data/mnist/t10k-labels-idx1-ubyte");
        
        auto conv1_w = load_txt("data/weights/conv1.weight.txt");
        auto conv1_b = load_txt("data/weights/conv1.bias.txt");
        auto conv2_w = load_txt("data/weights/conv2.weight.txt");
        auto conv2_b = load_txt("data/weights/conv2.bias.txt");
        auto fc1_w = load_txt("data/weights/fc1.weight.txt");
        auto fc1_b = load_txt("data/weights/fc1.bias.txt");
        auto fc2_w = load_txt("data/weights/fc2.weight.txt");
        auto fc2_b = load_txt("data/weights/fc2.bias.txt");
        auto fc3_w = load_txt("data/weights/fc3.weight.txt");
        auto fc3_b = load_txt("data/weights/fc3.bias.txt");
        
        std::cout << "    ✅ Données chargées avec succès" << std::endl;
        
        // ------------------------------------------------------------
        // 3. Génération des clés de rotation
        // ------------------------------------------------------------
        std::cout << "\n3. Génération des clés de rotation..." << std::endl;
        
        std::map<int, Ptr<ISwKey>> rot_keys;
        int max_rot = 900;  // Pour image 28×28 + décalages
        generate_all_rot_keys(*sk, max_rot, rot_keys);
        
        // ------------------------------------------------------------
        // 4. Inférence homomorphe sur N images
        // ------------------------------------------------------------
        std::cout << "\n4. Inférence homomorphe..." << std::endl;
        
        int num_test = std::min(10, (int)images.size());
        int correct = 0;
        
        for (int idx = 0; idx < num_test; ++idx) {
            std::cout << "\n--- Image " << idx << " ---" << std::endl;
            auto img_start = std::chrono::high_resolution_clock::now();
            
            // --------------------------------------------------------
            // Chiffrement de l'image (1×28×28)
            // --------------------------------------------------------
            std::cout << "    Chiffrement..." << std::endl;
            auto ct = encrypt_image(images[idx], *sk, encoder, encryptor);
            std::cout << "      Niveau initial: " << eval.getLevel(*ct) << std::endl;
            
            // --------------------------------------------------------
            // Conv1: 1×28×28 → 8×24×24
            // --------------------------------------------------------
            std::cout << "    Conv1..." << std::endl;
            ct = homomorphic_conv2d(
                *ct, conv1_w, conv1_b,
                1, 28, 28,     // in_c, in_h, in_w
                8, 5,          // out_c, kernel
                24, 24,        // out_h, out_w
                *sk, rot_keys, *relin_key, eval
            );
            
            // Bootstrap si nécessaire
            if (need_bootstrap(*ct, eval, 4)) {
                bootstrap_ciphertext(ct, *sk, eval);
            }
            
            // --------------------------------------------------------
            // ReLU1
            // --------------------------------------------------------
            std::cout << "    ReLU1..." << std::endl;
            double scale1 = 2.0;  // À ajuster selon la distribution
            ct = homomorphic_relu(*ct, 5, scale1, eval, *relin_key);
            
            // --------------------------------------------------------
            // AvgPool1: 8×24×24 → 8×12×12
            // --------------------------------------------------------
            std::cout << "    AvgPool1..." << std::endl;
            ct = homomorphic_avgpool2d(*ct, 8, 24, 24, rot_keys, eval);
            
            // --------------------------------------------------------
            // Conv2: 8×12×12 → 16×8×8
            // --------------------------------------------------------
            std::cout << "    Conv2..." << std::endl;
            ct = homomorphic_conv2d(
                *ct, conv2_w, conv2_b,
                8, 12, 12,     // in_c, in_h, in_w
                16, 5,         // out_c, kernel
                8, 8,          // out_h, out_w
                *sk, rot_keys, *relin_key, eval
            );
            
            // Bootstrap si nécessaire
            if (need_bootstrap(*ct, eval, 4)) {
                bootstrap_ciphertext(ct, *sk, eval);
            }
            
            // --------------------------------------------------------
            // ReLU2
            // --------------------------------------------------------
            std::cout << "    ReLU2..." << std::endl;
            double scale2 = 2.0;  // À ajuster
            ct = homomorphic_relu(*ct, 5, scale2, eval, *relin_key);
            
            // --------------------------------------------------------
            // AvgPool2: 16×8×8 → 16×4×4
            // --------------------------------------------------------
            std::cout << "    AvgPool2..." << std::endl;
            ct = homomorphic_avgpool2d(*ct, 16, 8, 8, rot_keys, eval);
            
            // --------------------------------------------------------
            // Flatten: 16×4×4 = 256
            // --------------------------------------------------------
            std::cout << "    Flatten..." << std::endl;
            // Pas d'opération FHE, juste réinterprétation des slots
            
            // --------------------------------------------------------
            // FC1: 256 → 128
            // --------------------------------------------------------
            std::cout << "    FC1..." << std::endl;
            ct = homomorphic_fc(
                *ct, fc1_w, fc1_b,
                256, 128, *sk, rot_keys, eval
            );
            
            // Bootstrap si nécessaire
            if (need_bootstrap(*ct, eval, 4)) {
                bootstrap_ciphertext(ct, *sk, eval);
            }
            
            // --------------------------------------------------------
            // ReLU3
            // --------------------------------------------------------
            std::cout << "    ReLU3..." << std::endl;
            double scale3 = 2.0;
            ct = homomorphic_relu(*ct, 5, scale3, eval, *relin_key);
            
            // --------------------------------------------------------
            // FC2: 128 → 64
            // --------------------------------------------------------
            std::cout << "    FC2..." << std::endl;
            ct = homomorphic_fc(
                *ct, fc2_w, fc2_b,
                128, 64, *sk, rot_keys, eval
            );
            
            // Bootstrap si nécessaire
            if (need_bootstrap(*ct, eval, 4)) {
                bootstrap_ciphertext(ct, *sk, eval);
            }
            
            // --------------------------------------------------------
            // ReLU4
            // --------------------------------------------------------
            std::cout << "    ReLU4..." << std::endl;
            double scale4 = 2.0;
            ct = homomorphic_relu(*ct, 5, scale4, eval, *relin_key);
            
            // --------------------------------------------------------
            // FC3: 64 → 10
            // --------------------------------------------------------
            std::cout << "    FC3..." << std::endl;
            auto ct_logits = homomorphic_fc(
                *ct, fc3_w, fc3_b,
                64, 10, *sk, rot_keys, eval
            );
            
            // --------------------------------------------------------
            // Déchiffrement et prédiction
            // --------------------------------------------------------
            std::cout << "    Déchiffrement..." << std::endl;
            auto logits = decrypt_result(*ct_logits, *sk, encoder, encryptor, 10);
            
            // Argmax
            int pred = 0;
            double max_val = logits[0];
            for (int i = 1; i < 10; ++i) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    pred = i;
                }
            }
            
            int true_label = labels[idx];
            bool is_correct = (pred == true_label);
            if (is_correct) correct++;
            
            auto img_end = std::chrono::high_resolution_clock::now();
            auto img_duration = std::chrono::duration_cast<std::chrono::milliseconds>(img_end - img_start);
            
            std::cout << "    Prédiction: " << pred 
                      << ", Vérité: " << true_label 
                      << " -> " << (is_correct ? "✅" : "❌") << std::endl;
            std::cout << "    Temps image: " << img_duration.count() << " ms" << std::endl;
            
            // Afficher les logits pour debug
            std::cout << "    Logits: [";
            for (int i = 0; i < 10; ++i) {
                std::cout << std::fixed << std::setprecision(2) << logits[i];
                if (i < 9) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        auto program_end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(program_end - program_start);
        auto avg_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            (program_end - program_start) / num_test
        );
        
        std::cout << "\n=== Résultats Finaux ===" << std::endl;
        std::cout << "  Images testées: " << num_test << std::endl;
        std::cout << "  Prédictions correctes: " << correct << std::endl;
        std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) 
                  << (100.0 * correct / num_test) << "%" << std::endl;
        std::cout << "  Temps total: " << total_time.count() << " s" << std::endl;
        std::cout << "  Temps moyen par image: " << avg_time.count() << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERREUR: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n✅ Pipeline terminé avec succès!" << std::endl;
    
    return 0;
}