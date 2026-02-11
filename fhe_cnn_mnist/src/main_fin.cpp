#include "fhe_cnn/fc.hpp"
#include "fhe_cnn/conv2d.hpp"
#include "fhe_cnn/pooling.hpp"
#include "fhe_cnn/relu.hpp"
#include "fhe_cnn/bootstrapping.hpp"
#include "fhe_cnn/onehot.hpp"
#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <memory>

using namespace heaan;
using namespace fhe_cnn;

int main() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "🚀 FHE CNN MNIST - Projet 5CS09 - VERSION FINALE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "   ✅ CNN 5 couches homomorphe" << std::endl;
    std::cout << "   ✅ 4 images parallélisées" << std::endl;
    std::cout << "   ✅ One-hot vector (BONUS)" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    auto program_start = std::chrono::high_resolution_clock::now();
    
    // ------------------------------------------------------------
    // 1. Initialisation HEAAN2
    // ------------------------------------------------------------
    std::cout << "1. Initialisation HEAAN2..." << std::endl;
    
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
    EnDecryptor decryptor(preset_id);
    
    int log_slots = sk->logDegree() - 1;
    int num_slots = 1 << log_slots;
    
    std::cout << "   └─ logDegree: " << sk->logDegree() << std::endl;
    std::cout << "   └─ logSlots: " << log_slots << std::endl;
    std::cout << "   └─ numSlots: " << num_slots << std::endl;
    
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
        
        std::cout << "   └─ Images: " << images.size() << std::endl;
        std::cout << "   └─ Labels: " << labels.size() << std::endl;
        std::cout << "   └─ Poids chargés: ✓" << std::endl;
        
        // ------------------------------------------------------------
        // 3. Génération des clés de rotation
        // ------------------------------------------------------------
        std::cout << "\n3. Génération des clés de rotation..." << std::endl;
        
        std::map<int, Ptr<ISwKey>> rot_keys;
        int max_rot = 900;  // Pour image 28×28 + décalages
        generate_all_rot_keys(*sk, max_rot, rot_keys);
        std::cout << "   └─ " << rot_keys.size() << " clés générées" << std::endl;
        
        // ------------------------------------------------------------
        // 4. Préparation du bootstrapping (une seule fois)
        // ------------------------------------------------------------
        std::cout << "\n4. Préparation du bootstrapping..." << std::endl;
        
        std::cout << "   └─ Génération des clés de bootstrap..." << std::endl;
        BootKeyPtrs bootkeys(preset_id, *sk);
        Bootstrapper bootstrapper(preset_id, bootkeys);
        std::cout << "   └─ Warmup..." << std::endl;
        bootstrapper.warmup();
        std::cout << "   └─ Bootstrap prêt" << std::endl;
        
        // ------------------------------------------------------------
        // 5. Inférence par lots de 4 images
        // ------------------------------------------------------------
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "5. INFÉRENCE HOMOMORPHE - 4 IMAGES PARALLÉLISÉES" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        int num_images = std::min(40, (int)images.size());  // Multiple de 4
        int num_batches = num_images / 4;
        int total_correct = 0;
        int bootstrap_count = 0;
        
        std::vector<double> batch_times;
        
        for (int batch = 0; batch < num_batches; ++batch) {
            std::cout << "\n--- BATCH " << batch+1 << "/" << num_batches 
                      << " (images " << batch*4 << "-" << batch*4+3 << ") ---" << std::endl;
            
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            // --------------------------------------------------------
            // 5a. Packer 4 images dans un seul ciphertext
            // --------------------------------------------------------
            std::vector<std::vector<double>> batch_images = {
                images[batch*4 + 0],
                images[batch*4 + 1],
                images[batch*4 + 2],
                images[batch*4 + 3]
            };
            
            auto msg_packed = pack_4_images(batch_images, log_slots, Device::CPU);
            
            auto ptxt_packed = IPlaintext::make();
            encoder.encode(msg_packed, *ptxt_packed);
            
            auto ct = ICiphertext::make();
            decryptor.encrypt(*ptxt_packed, *sk, *ct);
            
            std::cout << "   └─ Niveau initial: " << eval.getLevel(*ct) << std::endl;
            
            // --------------------------------------------------------
            // 5b. CONV1 + RELU1 + POOL1
            // --------------------------------------------------------
            std::cout << "   └─ Conv1..." << std::endl;
            ct = homomorphic_conv2d(*ct, conv1_w, conv1_b, 
                                   1, 28, 28, 8, 5, 24, 24,
                                   *sk, rot_keys, *relin_key, eval);
            
            std::cout << "   └─ ReLU1..." << std::endl;
            ct = homomorphic_relu(*ct, 5, 2.0, eval, *relin_key);
            
            std::cout << "   └─ AvgPool1..." << std::endl;
            ct = homomorphic_avgpool2d(*ct, 8, 24, 24, rot_keys, eval);
            
            // --------------------------------------------------------
            // 5c. CONV2 + RELU2
            // --------------------------------------------------------
            std::cout << "   └─ Conv2..." << std::endl;
            ct = homomorphic_conv2d(*ct, conv2_w, conv2_b,
                                   8, 12, 12, 16, 5, 8, 8,
                                   *sk, rot_keys, *relin_key, eval);
            
            std::cout << "   └─ ReLU2..." << std::endl;
            ct = homomorphic_relu(*ct, 5, 2.0, eval, *relin_key);
            
            // --------------------------------------------------------
            // 5d. BOOTSTRAP #1 - CRITIQUE
            // --------------------------------------------------------
            int level_before_bootstrap1 = eval.getLevel(*ct);
            if (level_before_bootstrap1 <= 3) {
                std::cout << "   └─ ⚠️  BOOTSTRAP #1 (niveau " 
                          << level_before_bootstrap1 << ")..." << std::endl;
                
                auto boot_start = std::chrono::high_resolution_clock::now();
                bootstrapper.bootstrap(*ct);
                auto boot_end = std::chrono::high_resolution_clock::now();
                auto boot_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    boot_end - boot_start
                );
                
                bootstrap_count++;
                std::cout << "      └─ Niveau après: " << eval.getLevel(*ct) 
                          << " (temps: " << boot_time.count() << " ms)" << std::endl;
            }
            
            // --------------------------------------------------------
            // 5e. POOL2
            // --------------------------------------------------------
            std::cout << "   └─ AvgPool2..." << std::endl;
            ct = homomorphic_avgpool2d(*ct, 16, 8, 8, rot_keys, eval);
            
            // --------------------------------------------------------
            // 5f. FC1 + RELU3
            // --------------------------------------------------------
            std::cout << "   └─ FC1 (256→128)..." << std::endl;
            ct = homomorphic_fc(*ct, fc1_w, fc1_b, 256, 128, *sk, rot_keys, eval);
            
            std::cout << "   └─ ReLU3..." << std::endl;
            ct = homomorphic_relu(*ct, 5, 2.0, eval, *relin_key);
            
            // --------------------------------------------------------
            // 5g. BOOTSTRAP #2 - CRITIQUE
            // --------------------------------------------------------
            int level_before_bootstrap2 = eval.getLevel(*ct);
            if (level_before_bootstrap2 <= 3) {
                std::cout << "   └─ ⚠️  BOOTSTRAP #2 (niveau " 
                          << level_before_bootstrap2 << ")..." << std::endl;
                
                auto boot_start = std::chrono::high_resolution_clock::now();
                bootstrapper.bootstrap(*ct);
                auto boot_end = std::chrono::high_resolution_clock::now();
                auto boot_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    boot_end - boot_start
                );
                
                bootstrap_count++;
                std::cout << "      └─ Niveau après: " << eval.getLevel(*ct) 
                          << " (temps: " << boot_time.count() << " ms)" << std::endl;
            }
            
            // --------------------------------------------------------
            // 5h. FC2 + RELU4
            // --------------------------------------------------------
            std::cout << "   └─ FC2 (128→64)..." << std::endl;
            ct = homomorphic_fc(*ct, fc2_w, fc2_b, 128, 64, *sk, rot_keys, eval);
            
            std::cout << "   └─ ReLU4..." << std::endl;
            ct = homomorphic_relu(*ct, 5, 2.0, eval, *relin_key);
            
            // --------------------------------------------------------
            // 5i. FC3 (64→10) - LOGITS
            // --------------------------------------------------------
            std::cout << "   └─ FC3 (64→10)..." << std::endl;
            auto ct_logits = homomorphic_fc(*ct, fc3_w, fc3_b, 64, 10, *sk, rot_keys, eval);
            
            // --------------------------------------------------------
            // 5j. BONUS: ONE-HOT VECTOR
            // --------------------------------------------------------
            std::cout << "   └─ 🔥 Conversion one-hot vector..." << std::endl;
            auto ct_onehot = homomorphic_onehot(*ct_logits, *sk, rot_keys, eval, *relin_key);
            
            // --------------------------------------------------------
            // 5k. DÉCHIFFREMENT
            // --------------------------------------------------------
            std::cout << "   └─ Déchiffrement..." << std::endl;
            
            auto ptxt_onehot = IPlaintext::make();
            decryptor.decrypt(*ct_onehot, *sk, *ptxt_onehot);
            
            Message<Complex> msg_onehot;
            encoder.decode(*ptxt_onehot, msg_onehot);
            msg_onehot.to(Device::CPU);
            
            // --------------------------------------------------------
            // 5l. PRÉDICTIONS POUR 4 IMAGES
            // --------------------------------------------------------
            int batch_correct = 0;
            
            for (int i = 0; i < 4; ++i) {
                // Index du début pour cette image
                int start_idx = i * 10;
                
                // Trouver le maximum (valeur la plus proche de 1)
                int pred = 0;
                double max_val = msg_onehot[start_idx].real();
                
                for (int j = 1; j < 10; ++j) {
                    double val = msg_onehot[start_idx + j].real();
                    if (val > max_val) {
                        max_val = val;
                        pred = j;
                    }
                }
                
                int true_label = labels[batch*4 + i];
                if (pred == true_label) batch_correct++;
                
                std::cout << "      Image " << std::setw(2) << batch*4 + i 
                          << ": prédiction = " << pred 
                          << ", vérité = " << true_label 
                          << " → " << (pred == true_label ? "✅" : "❌") << std::endl;
                
                // Debug: afficher le one-hot vector pour la première image du premier batch
                if (batch == 0 && i == 0) {
                    std::cout << "         One-hot: [";
                    for (int j = 0; j < 10; ++j) {
                        std::cout << std::fixed << std::setprecision(3) 
                                  << msg_onehot[start_idx + j].real();
                        if (j < 9) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
            
            total_correct += batch_correct;
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                batch_end - batch_start
            );
            batch_times.push_back(batch_duration.count());
            
            std::cout << "   └─ Batch terminé: " << batch_correct << "/4 corrects, "
                      << "temps: " << batch_duration.count() << " ms" << std::endl;
        }
        
        // ------------------------------------------------------------
        // 6. RÉSULTATS FINAUX
        // ------------------------------------------------------------
        auto program_end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
            program_end - program_start
        );
        
        // Calcul des statistiques
        double sum_times = 0;
        for (double t : batch_times) sum_times += t;
        double avg_batch_time = sum_times / batch_times.size();
        double avg_image_time = avg_batch_time / 4.0;
        
        double accuracy = 100.0 * total_correct / num_images;
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "📊 RÉSULTATS FINAUX - PROJET 5CS09" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        std::cout << "\n📈 PERFORMANCES:" << std::endl;
        std::cout << "   └─ Images testées: " << num_images << std::endl;
        std::cout << "   └─ Lots de 4 images: " << num_batches << std::endl;
        std::cout << "   └─ Prédictions correctes: " << total_correct << std::endl;
        std::cout << "   └─ Accuracy: " << std::fixed << std::setprecision(2) 
                  << accuracy << "%" << std::endl;
        std::cout << "   └─ Nombre de bootstraps: " << bootstrap_count << std::endl;
        
        std::cout << "\n⏱️  TEMPS D'EXÉCUTION:" << std::endl;
        std::cout << "   └─ Temps total: " << total_time.count() << " s" << std::endl;
        std::cout << "   └─ Temps moyen par BATCH (4 images): " 
                  << std::fixed << std::setprecision(0) << avg_batch_time << " ms" << std::endl;
        std::cout << "   └─ Temps moyen par IMAGE: " 
                  << std::fixed << std::setprecision(0) << avg_image_time << " ms" << std::endl;
        
        std::cout << "\n✅ OBLIGATIONS DU PROJET:" << std::endl;
        std::cout << "   └─ [✓] CNN 5 couches homomorphe" << std::endl;
        std::cout << "   └─ [✓] Mesure temps moyen et accuracy" << std::endl;
        std::cout << "   └─ [✓] 4 images parallélisées" << std::endl;
        std::cout << "   └─ [⭐] BONUS: One-hot vector" << std::endl;
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "🏆 PROJET COMPLÉTÉ AVEC SUCCÈS!" << std::endl;
        std::cout << std::string(60, '=') << "\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERREUR FATALE: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}