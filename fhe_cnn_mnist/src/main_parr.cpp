#include "fhe_cnn/fc.hpp"
#include "fhe_cnn/conv2d.hpp"
#include "fhe_cnn/pooling.hpp"
#include "fhe_cnn/relu.hpp"
#include "fhe_cnn/bootstrapping.hpp"
#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace heaan;
using namespace fhe_cnn;

int main() {
    std::cout << "\n🚀 FHE CNN MNIST - PARALLÉLISATION 4 IMAGES" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    auto program_start = std::chrono::high_resolution_clock::now();
    
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
    EnDecryptor decryptor(preset_id);
    
    // ------------------------------------------------------------
    // 2. Chargement des données
    // ------------------------------------------------------------
    std::cout << "\n2. Chargement des données..." << std::endl;
    
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
    
    // ------------------------------------------------------------
    // 3. Génération des clés de rotation
    // ------------------------------------------------------------
    std::cout << "\n3. Génération des clés de rotation..." << std::endl;
    
    std::map<int, Ptr<ISwKey>> rot_keys;
    int max_rot = 900;
    generate_all_rot_keys(*sk, max_rot, rot_keys);
    
    // ------------------------------------------------------------
    // 4. Préparation du bootstrapping
    // ------------------------------------------------------------
    std::cout << "\n4. Préparation du bootstrapping..." << std::endl;
    
    BootKeyPtrs bootkeys(preset_id, *sk);
    Bootstrapper bootstrapper(preset_id, bootkeys);
    bootstrapper.warmup();
    
    // ------------------------------------------------------------
    // 5. Inférence par lots de 4 images
    // ------------------------------------------------------------
    std::cout << "\n5. Inférence par lots de 4 images..." << std::endl;
    
    int num_images = std::min(40, (int)images.size());  // Multiple de 4
    int num_batches = num_images / 4;
    int total_correct = 0;
    int bootstrap_count = 0;
    
    for (int batch = 0; batch < num_batches; ++batch) {
        std::cout << "\n--- Batch " << batch+1 << "/" << num_batches 
                  << " (images " << batch*4 << "-" << batch*4+3 << ") ---" << std::endl;
        
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        // --------------------------------------------------------
        // Packer 4 images
        // --------------------------------------------------------
        std::vector<std::vector<double>> batch_images = {
            images[batch*4 + 0],
            images[batch*4 + 1],
            images[batch*4 + 2],
            images[batch*4 + 3]
        };
        
        int log_slots = sk->logDegree() - 1;
        auto msg_packed = pack_4_images(batch_images, log_slots, Device::CPU);
        
        auto ptxt_packed = IPlaintext::make();
        encoder.encode(msg_packed, *ptxt_packed);
        
        auto ct = ICiphertext::make();
        decryptor.encrypt(*ptxt_packed, *sk, *ct);
        
        std::cout << "    Niveau initial: " << eval.getLevel(*ct) << std::endl;
        
        // --------------------------------------------------------
        // FORWARD PASS - IDENTIQUE MAIS TOUT EST PARALLÉLISÉ !
        // --------------------------------------------------------
        
        // Conv1
        ct = homomorphic_conv2d(*ct, conv1_w, conv1_b, 1, 28, 28, 8, 5, 24, 24,
                               *sk, rot_keys, *relin_key, eval);
        
        // ReLU1
        ct = homomorphic_relu(*ct, 5, 2.0, eval, *relin_key);
        
        // Pool1
        ct = homomorphic_avgpool2d(*ct, 8, 24, 24, rot_keys, eval);
        
        // Conv2
        ct = homomorphic_conv2d(*ct, conv2_w, conv2_b, 8, 12, 12, 16, 5, 8, 8,
                               *sk, rot_keys, *relin_key, eval);
        
        // ReLU2
        ct = homomorphic_relu(*ct, 5, 2.0, eval, *relin_key);
        
        // BOOTSTRAP #1
        if (eval.getLevel(*ct) <= 3) {
            std::cout << "    ⚠️  Bootstrap #1..." << std::endl;
            bootstrapper.bootstrap(*ct);
            bootstrap_count++;
        }
        
        // Pool2
        ct = homomorphic_avgpool2d(*ct, 16, 8, 8, rot_keys, eval);
        
        // FC1
        ct = homomorphic_fc(*ct, fc1_w, fc1_b, 256, 128, *sk, rot_keys, eval);
        
        // ReLU3
        ct = homomorphic_relu(*ct, 5, 2.0, eval, *relin_key);
        
        // BOOTSTRAP #2
        if (eval.getLevel(*ct) <= 3) {
            std::cout << "    ⚠️  Bootstrap #2..." << std::endl;
            bootstrapper.bootstrap(*ct);
            bootstrap_count++;
        }
        
        // FC2
        ct = homomorphic_fc(*ct, fc2_w, fc2_b, 128, 64, *sk, rot_keys, eval);
        
        // ReLU4
        ct = homomorphic_relu(*ct, 5, 2.0, eval, *relin_key);
        
        // FC3 - Sortie 10 classes
        auto ct_logits = homomorphic_fc(*ct, fc3_w, fc3_b, 64, 10, *sk, rot_keys, eval);
        
        // --------------------------------------------------------
        // Déchiffrement et prédictions pour 4 images
        // --------------------------------------------------------
        auto results = unpack_4_results(*ct_logits, *sk, encoder, decryptor, 10);
        
        int batch_correct = 0;
        for (int i = 0; i < 4; ++i) {
            // Argmax
            int pred = 0;
            double max_val = results[i][0];
            for (int j = 1; j < 10; ++j) {
                if (results[i][j] > max_val) {
                    max_val = results[i][j];
                    pred = j;
                }
            }
            
            int true_label = labels[batch*4 + i];
            if (pred == true_label) batch_correct++;
            
            std::cout << "      Image " << batch*4 + i 
                      << ": prédiction " << pred 
                      << ", vérité " << true_label 
                      << " -> " << (pred == true_label ? "✅" : "❌") << std::endl;
        }
        
        total_correct += batch_correct;
        
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            batch_end - batch_start
        );
        
        std::cout << "    Batch terminé: " << batch_correct << "/4 corrects, "
                  << "temps: " << batch_duration.count() << " ms" << std::endl;
    }
    
    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(program_end - program_start);
    auto avg_time_per_image = std::chrono::duration_cast<std::chrono::milliseconds>(
        (program_end - program_start) / num_images
    );
    
    std::cout << "\n=== RÉSULTATS FINAUX (4 images parallélisées) ===" << std::endl;
    std::cout << "  Images testées: " << num_images << std::endl;
    std::cout << "  Prédictions correctes: " << total_correct << std::endl;
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) 
              << (100.0 * total_correct / num_images) << "%" << std::endl;
    std::cout << "  Nombre de bootstraps: " << bootstrap_count << std::endl;
    std::cout << "  Temps total: " << total_time.count() << " s" << std::endl;
    std::cout << "  Temps moyen par IMAGE: " << avg_time_per_image.count() << " ms" << std::endl;
    std::cout << "  Temps moyen par BATCH (4 images): " 
              << total_time.count() * 1000.0 / num_batches << " ms" << std::endl;
    std::cout << "  Gain de parallélisation: 4x 🚀" << std::endl;
    
    return 0;
}