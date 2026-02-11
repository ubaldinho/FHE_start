#include "fhe_cnn/bootstrapping.hpp"
#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <chrono>

using namespace heaan;
using namespace fhe_cnn;

int main() {
    std::cout << "\n🧪 Test Bootstrapping" << std::endl;
    std::cout << "=====================" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // ------------------------------------------------------------
    // 1. Initialisation HEAAN2
    // ------------------------------------------------------------
    std::cout << "\n1. Initialisation HEAAN2..." << std::endl;
    
    auto preset_id = PresetParamsId::F16Opt_Gr;
    
    SKGenerator skgen(preset_id);
    auto sk = skgen.genKey();
    sk->to(Device::CPU);
    
    HomEval eval(preset_id);
    EnDecoder encoder(preset_id);
    EnDecryptor encryptor(preset_id);
    
    // ------------------------------------------------------------
    // 2. Créer un ciphertext simple
    // ------------------------------------------------------------
    std::cout << "\n2. Création d'un ciphertext..." << std::endl;
    
    int log_slots = sk->logDegree() - 1;
    Message<Complex> msg(log_slots, Device::CPU);
    msg[0] = Complex(3.14159, 0.0);  // π dans le slot 0
    for (int i = 1; i < (1 << log_slots); ++i) {
        msg[i] = Complex(0.0, 0.0);
    }
    
    auto ptxt = IPlaintext::make();
    encoder.encode(msg, *ptxt);
    
    auto ctxt = ICiphertext::make();
    encryptor.encrypt(*ptxt, *sk, *ctxt);
    
    // ------------------------------------------------------------
    // 3. Consommer des niveaux volontairement
    // ------------------------------------------------------------
    std::cout << "\n3. Consommation de niveaux..." << std::endl;
    
    for (int i = 0; i < 8; ++i) {
        // Multiplier par 1.0 pour consommer un niveau
        auto ct_mul = ICiphertext::make();
        eval.mul(*ctxt, 1.0, *ct_mul);
        eval.rescale(*ct_mul, *ct_mul);
        ctxt = std::move(ct_mul);
        
        std::cout << "    Niveau après multiplication " << i+1 
                  << ": " << eval.getLevel(*ctxt) << std::endl;
    }
    
    // ------------------------------------------------------------
    // 4. Vérifier si bootstrap nécessaire
    // ------------------------------------------------------------
    std::cout << "\n4. Vérification besoin bootstrap..." << std::endl;
    
    if (need_bootstrap(*ctxt, eval, 3)) {
        std::cout << "    ✅ Bootstrap nécessaire" << std::endl;
    } else {
        std::cout << "    Bootstrap non nécessaire" << std::endl;
    }
    
    // ------------------------------------------------------------
    // 5. Exécuter le bootstrap
    // ------------------------------------------------------------
    std::cout << "\n5. Exécution du bootstrapping..." << std::endl;
    
    bootstrap_ciphertext(ctxt, *sk, eval, preset_id);
    
    // ------------------------------------------------------------
    // 6. Déchiffrer et vérifier
    // ------------------------------------------------------------
    std::cout << "\n6. Vérification du résultat..." << std::endl;
    
    auto ptxt_result = IPlaintext::make();
    encryptor.decrypt(*ctxt, *sk, *ptxt_result);
    
    Message<Complex> msg_result;
    encoder.decode(*ptxt_result, msg_result);
    msg_result.to(Device::CPU);
    
    double value = msg_result[0].real();
    double expected = 3.14159;
    double err = std::abs(value - expected);
    
    std::cout << "    Valeur attendue: " << expected << std::endl;
    std::cout << "    Valeur après bootstrap: " << value << std::endl;
    std::cout << "    Erreur: " << err << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n=== Statistiques ===" << std::endl;
    std::cout << "  Erreur: " << err << std::endl;
    std::cout << "  Erreur (log2): " << std::log2(err) << " bits" << std::endl;
    std::cout << "  Temps total: " << duration.count() << " ms" << std::endl;
    
    if (err < 1e-5) {
        std::cout << "\n✅ TEST PASSÉ!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ TEST ÉCHOUÉ!" << std::endl;
        return 1;
    }
}