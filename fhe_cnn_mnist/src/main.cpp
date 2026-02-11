#include "fhe_cnn/fc.hpp"
#include "fhe_cnn/conv2d.hpp"
#include "fhe_cnn/pooling.hpp"
#include "fhe_cnn/relu.hpp"
#include "fhe_cnn/bootstrapping.hpp"
#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <chrono>
#include <memory>

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
    
    SKGenerator skgen(preset_id);
    auto sk = skgen.genKey();
    sk->to(Device::CPU);
    
    SwKeyGenerator swkgen(preset_id);
    auto relin_key = swkgen.genRelinKey(*sk);
    
    HomEval eval(preset_id);
    EnDecoder encoder(preset_id);
    EnDecryptor encryptor(preset_id);
    
    std::cout << "    logDegree: " << sk->logDegree() << std::endl;
    std::cout << "    logSlots: " << sk->logDegree() - 1 << std::endl;
    
    // ------------------------------------------------------------
    // 2. Chargement des données MNIST et poids
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
    int max_rot = 256;  // Pour FC et Conv
    generate_all_rot_keys(*sk, max_rot, rot_keys);
    
    // ------------------------------------------------------------
    // 4. Pipeline d'inférence homomorphe
    // ------------------------------------------------------------
    std::cout << "\n4. Inférence homomorphe..." << std::endl;
    
    int num_test = std::min(10, (int)images.size());
    int correct = 0;
    
    for (int idx = 0; idx < num_test; ++idx) {
        std::cout << "\n--- Image " << idx << " ---" << std::endl;
        
        // --------------------------------------------------------
        // Chiffrement de l'image
        // --------------------------------------------------------
        auto ct = encrypt_image(images[idx], *sk, encoder, encryptor);
        
        // --------------------------------------------------------
        // Conv1 + ReLU + Pool1
        // --------------------------------------------------------
        // TODO: Implémenter convolution
        // TODO: Implémenter ReLU
        // TODO: Implémenter AveragePool
        
        // --------------------------------------------------------
        // Conv2 + ReLU + Pool2
        // --------------------------------------------------------
        // TODO: ...
        
        // --------------------------------------------------------
        // Flatten + FC1 + ReLU
        // --------------------------------------------------------
        // TODO: ...
        
        // --------------------------------------------------------
        // FC2 + ReLU
        // --------------------------------------------------------
        // TODO: ...
        
        // --------------------------------------------------------
        // FC3 (sortie 10 classes)
        // --------------------------------------------------------
        // TODO: ...
        
        // --------------------------------------------------------
        // Déchiffrement et prédiction
        // --------------------------------------------------------
        // TODO: ...
        
        std::cout << "    Image " << idx << " traitée" << std::endl;
    }
    
    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(program_end - program_start);
    
    std::cout << "\n=== Résultats ===" << std::endl;
    std::cout << "  Images testées: " << num_test << std::endl;
    std::cout << "  Prédictions correctes: " << correct << std::endl;
    std::cout << "  Accuracy: " << 100.0 * correct / num_test << "%" << std::endl;
    std::cout << "  Temps total: " << total_time.count() << " s" << std::endl;
    std::cout << "  Temps par image: " << total_time.count() * 1000.0 / num_test << " ms" << std::endl;
    
    std::cout << "\n✅ Projet terminé!" << std::endl;
    
    return 0;
}