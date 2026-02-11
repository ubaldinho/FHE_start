#include "fhe_cnn/onehot.hpp"
#include "fhe_cnn/utils.hpp"
#include "fhe_cnn/bootstrapping.hpp"
#include <iostream>
#include <chrono>
#include <random>

using namespace heaan;
using namespace fhe_cnn;

int main() {
    std::cout << "\n🧪 Test One-hot Vector" << std::endl;
    std::cout << "======================" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // ------------------------------------------------------------
    // 1. Initialisation
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
    // 2. Clés de rotation
    // ------------------------------------------------------------
    std::cout << "\n2. Génération des clés..." << std::endl;
    
    std::map<int, Ptr<ISwKey>> rot_keys;
    for (int rot : {1,2,3,4,5,6,7,8,9}) {
        auto rot_key = swkgen.genRotKey(*sk, rot);
        rot_keys[rot] = std::move(rot_key);
    }
    for (int shift = 1; shift < 16; shift <<= 1) {
        if (rot_keys.find(shift) == rot_keys.end()) {
            auto rot_key = swkgen.genRotKey(*sk, shift);
            rot_keys[shift] = std::move(rot_key);
        }
    }
    
    // ------------------------------------------------------------
    // 3. Création des logits de test
    // ------------------------------------------------------------
    std::cout << "\n3. Création des logits de test..." << std::endl;
    
    std::vector<double> logits = {
        0.1, 0.5, 0.3, 0.8, 0.2,  // Max à 0.8 (index 3)
        0.4, 0.6, 0.7, 0.9, 0.0   // Max à 0.9 (index 8)
    };
    // Correction: mettons le max à 0.9 à l'index 8
    logits[8] = 0.9;  // S'assurer que c'est le max
    
    int log_slots = sk->logDegree() - 1;
    Message<Complex> msg_logits(log_slots, Device::CPU);
    
    for (int i = 0; i < 10; ++i) {
        msg_logits[i] = Complex(logits[i], 0.0);
    }
    for (int i = 10; i < (1 << log_slots); ++i) {
        msg_logits[i] = Complex(0.0, 0.0);
    }
    
    auto ptxt_logits = IPlaintext::make();
    encoder.encode(msg_logits, *ptxt_logits);
    
    auto ct_logits = ICiphertext::make();
    decryptor.encrypt(*ptxt_logits, *sk, *ct_logits);
    
    std::cout << "    Logits: [";
    for (int i = 0; i < 10; ++i) {
        std::cout << logits[i];
        if (i < 9) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "    Max attendu: index 8 (0.9)" << std::endl;
    
    // ------------------------------------------------------------
    // 4. One-hot homomorphique
    // ------------------------------------------------------------
    std::cout << "\n4. Exécution one-hot..." << std::endl;
    
    auto ct_onehot = homomorphic_onehot(*ct_logits, *sk, rot_keys, eval, *relin_key);
    
    // ------------------------------------------------------------
    // 5. Déchiffrement
    // ------------------------------------------------------------
    std::cout << "\n5. Déchiffrement..." << std::endl;
    
    auto ptxt_onehot = IPlaintext::make();
    decryptor.decrypt(*ct_onehot, *sk, *ptxt_onehot);
    
    Message<Complex> msg_onehot;
    encoder.decode(*ptxt_onehot, msg_onehot);
    msg_onehot.to(Device::CPU);
    
    // ------------------------------------------------------------
    // 6. Vérification
    // ------------------------------------------------------------
    std::cout << "\n6. Vérification..." << std::endl;
    
    std::cout << "    One-hot vector: [";
    int max_index = 0;
    double max_val = msg_onehot[0].real();
    
    for (int i = 0; i < 10; ++i) {
        double val = msg_onehot[i].real();
        std::cout << std::fixed << std::setprecision(3) << val;
        if (i < 9) std::cout << ", ";
        if (val > max_val) {
            max_val = val;
            max_index = i;
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "    Index du max: " << max_index << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n=== Résultats ===" << std::endl;
    std::cout << "  Index prédit: " << max_index << std::endl;
    std::cout << "  Index attendu: 8" << std::endl;
    std::cout << "  ✓ One-hot vector généré" << std::endl;
    std::cout << "  Temps: " << duration.count() << " ms" << std::endl;
    
    if (max_index == 8) {
        std::cout << "\n✅ TEST PASSÉ!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ TEST ÉCHOUÉ!" << std::endl;
        return 1;
    }
}