#include "fhe_cnn/bootstrapping.hpp"
#include <iostream>
#include <chrono>

namespace fhe_cnn {

using namespace heaan;

bool need_bootstrap(
    const ICiphertext& ctxt,
    HomEval& eval,
    int threshold
) {
    try {
        int level = eval.getLevel(ctxt);
        if (level <= threshold) {
            std::cout << "⚠️  Niveau bas: " << level 
                      << " (seuil: " << threshold << ")" << std::endl;
            return true;
        }
        return false;
    } catch (const std::exception& e) {
        std::cerr << "  ERREUR need_bootstrap: " << e.what() << std::endl;
        return true;  // En cas d'erreur, on bootstrap par sécurité
    }
}

void bootstrap_ciphertext(
    Ptr<ICiphertext>& ctxt,
    const ISecretKey& sk,
    HomEval& eval,
    PresetParamsId preset_id
) {
    std::cout << "🔷 Bootstrapping..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        // --------------------------------------------------------
        // 1. Vérifier le niveau avant bootstrap
        // --------------------------------------------------------
        int level_before = eval.getLevel(*ctxt);
        std::cout << "    Niveau avant: " << level_before << std::endl;
        
        // --------------------------------------------------------
        // 2. Créer les clés de bootstrapping
        // --------------------------------------------------------
        std::cout << "    Génération des clés de bootstrap..." << std::endl;
        BootKeyPtrs bootkeys(preset_id, sk);
        
        // --------------------------------------------------------
        // 3. Initialiser le bootstrapper
        // --------------------------------------------------------
        Bootstrapper bootstrapper(preset_id, bootkeys);
        
        // --------------------------------------------------------
        // 4. Warmup (optionnel mais recommandé)
        // --------------------------------------------------------
        std::cout << "    Warmup..." << std::endl;
        bootstrapper.warmup();
        
        // --------------------------------------------------------
        // 5. Bootstrapper le ciphertext
        // --------------------------------------------------------
        std::cout << "    Bootstrap en cours..." << std::endl;
        bootstrapper.bootstrap(*ctxt);
        
        // --------------------------------------------------------
        // 6. Vérifier le niveau après bootstrap
        // --------------------------------------------------------
        int level_after = eval.getLevel(*ctxt);
        std::cout << "    Niveau après: " << level_after << std::endl;
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "    ✅ Bootstrapping réussi en " 
                  << duration.count() << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "    ❌ ERREUR Bootstrapping: " << e.what() << std::endl;
        throw;
    }
}

} // namespace fhe_cnn