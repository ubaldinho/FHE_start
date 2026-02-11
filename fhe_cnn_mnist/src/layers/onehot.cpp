#include "fhe_cnn/onehot.hpp"
#include "fhe_cnn/bootstrapping.hpp"
#include <iostream>
#include <cmath>

namespace fhe_cnn {

using namespace heaan;

// ------------------------------------------------------------
// Approximation de la fonction de Heaviside (x > 0)
// Polynomiale degré 3: 0.5 + 0.5 * (x / (1 + |x|))
// ------------------------------------------------------------
Ptr<ICiphertext> homomorphic_gt(
    const ICiphertext& x_enc,
    const ICiphertext& y_enc,
    HomEval& eval,
    const ISwKey& relin_key
) {
    // x - y
    auto ct_diff = ICiphertext::make();
    eval.sub(x_enc, y_enc, *ct_diff);
    
    // Approximation de sign(x) sur [-1, 1]
    // On utilise: 0.5 + 0.5 * (x / (1 + |x|))
    // Approximation polynomiale: 0.5 + 0.3125x - 0.0625x^3
    
    // |x| ≈ x * sign(x) -> approximation directe
    
    // Version simplifiée: fonction signe approximée
    // sign(x) ≈ 0.5 + 0.5x - 0.125x^3 pour x dans [-2,2]
    
    auto ct_x = ICiphertext::make();
    *ct_x = ct_diff;
    
    // Mise à l'échelle pour être dans [-1,1]
    eval.mul(*ct_x, 0.5, *ct_x);
    eval.rescale(*ct_x, *ct_x);
    
    // x^3
    auto ct_x2 = ICiphertext::make();
    eval.tensor(*ct_x, *ct_x, *ct_x2);
    eval.relin(*ct_x2, relin_key);
    eval.rescale(*ct_x2, *ct_x2);
    
    auto ct_x3 = ICiphertext::make();
    eval.tensor(*ct_x2, *ct_x, *ct_x3);
    eval.relin(*ct_x3, relin_key);
    eval.rescale(*ct_x3, *ct_x3);
    
    // 0.5x
    auto ct_term1 = ICiphertext::make();
    eval.mul(*ct_x, 0.5, *ct_term1);
    eval.rescale(*ct_term1, *ct_term1);
    
    // 0.125x^3
    auto ct_term2 = ICiphertext::make();
    eval.mul(*ct_x3, 0.125, *ct_term2);
    eval.rescale(*ct_term2, *ct_term2);
    
    // 0.5 + 0.5x - 0.125x^3
    auto ct_sign = ICiphertext::make();
    eval.add(*ct_term1, *ct_term2, *ct_sign);
    eval.add(*ct_sign, 0.5, *ct_sign);
    
    return ct_sign;
}

// ------------------------------------------------------------
// Trouver le maximum par tournoi binaire
// ------------------------------------------------------------
Ptr<ICiphertext> homomorphic_max(
    const ICiphertext& logits_enc,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval,
    const ISwKey& relin_key
) {
    std::cout << "    🔍 Recherche du maximum..." << std::endl;
    
    auto ct_current = ICiphertext::make();
    *ct_current = logits_enc;
    
    // Tournoi à 10 éléments (4 rounds)
    std::vector<int> rounds = {1, 2, 4, 8};
    int total_rounds = 0;
    
    for (int shift : rounds) {
        if (shift >= 10) break;
        
        // Rotation
        auto ct_rot = ICiphertext::make();
        auto it = rot_keys.find(shift);
        if (it == rot_keys.end()) {
            std::cerr << "    ERREUR: Clé rotation " << shift << " manquante!" << std::endl;
            continue;
        }
        eval.rot(*ct_current, shift, *ct_rot, *(it->second));
        
        // Comparaison: max(x, y) = x + (y-x)*gt(y,x)
        auto ct_gt = homomorphic_gt(*ct_rot, *ct_current, eval, relin_key);
        
        // (y - x) * gt(y,x)
        auto ct_diff = ICiphertext::make();
        eval.sub(*ct_rot, *ct_current, *ct_diff);
        
        auto ct_mult = ICiphertext::make();
        eval.mul(*ct_diff, *ct_gt, *ct_mult);
        eval.rescale(*ct_mult, *ct_mult);
        
        // x + (y-x)*gt(y,x) = max(x,y)
        auto ct_max = ICiphertext::make();
        eval.add(*ct_current, *ct_mult, *ct_max);
        *ct_current = *ct_max;
        
        total_rounds++;
        std::cout << "      Round " << total_rounds << " (shift=" << shift << ") OK" << std::endl;
    }
    
    return ct_current;
}

// ------------------------------------------------------------
// One-hot vector complet
// ------------------------------------------------------------
Ptr<ICiphertext> homomorphic_onehot(
    const ICiphertext& logits_enc,
    const ISecretKey& sk,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval,
    const ISwKey& relin_key
) {
    std::cout << "    🔥 Conversion en one-hot vector..." << std::endl;
    
    // --------------------------------------------------------
    // 1. Trouver la valeur maximale
    // --------------------------------------------------------
    auto ct_max = homomorphic_max(logits_enc, rot_keys, eval, relin_key);
    
    // Bootstrap si nécessaire
    if (eval.getLevel(*ct_max) < 5) {
        std::cout << "      Bootstrap pour one-hot..." << std::endl;
        BootKeyPtrs bootkeys(PresetParamsId::F16Opt_Gr, sk);
        Bootstrapper bootstrapper(PresetParamsId::F16Opt_Gr, bootkeys);
        bootstrapper.warmup();
        bootstrapper.bootstrap(*ct_max);
    }
    
    // --------------------------------------------------------
    // 2. Créer le masque pour duplication du max
    // --------------------------------------------------------
    int log_slots = sk.logDegree() - 1;
    EnDecoder encoder(PresetParamsId::F16Opt_Gr);
    
    Message<Complex> msg_mask(log_slots, Device::CPU);
    msg_mask[0] = Complex(1.0, 0.0);
    for (int i = 1; i < (1 << log_slots); ++i) {
        msg_mask[i] = Complex(0.0, 0.0);
    }
    
    auto ptxt_mask = IPlaintext::make();
    encoder.encode(msg_mask, *ptxt_mask);
    
    // Extraire le max du slot 0
    auto ct_max_slot0 = ICiphertext::make();
    auto ptxt_mask_leveled = IPlaintext::make();
    eval.levelDownTo(*ptxt_mask, *ptxt_mask_leveled, eval.getLevel(*ct_max));
    eval.mul(*ct_max, *ptxt_mask_leveled, *ct_max_slot0);
    eval.rescale(*ct_max_slot0, *ct_max_slot0);
    
    // Dupliquer dans tous les slots 0-9
    auto ct_max_all = ICiphertext::make();
    *ct_max_all = *ct_max_slot0;
    
    for (int i = 1; i < 10; ++i) {
        auto ct_rot = ICiphertext::make();
        auto it = rot_keys.find(i);
        if (it != rot_keys.end()) {
            eval.rot(*ct_max_slot0, i, *ct_rot, *(it->second));
            
            auto ct_add = ICiphertext::make();
            eval.add(*ct_max_all, *ct_rot, *ct_add);
            *ct_max_all = *ct_add;
        }
    }
    
    // --------------------------------------------------------
    // 3. Comparer chaque logit avec le max
    // --------------------------------------------------------
    auto ct_onehot = ICiphertext::make();
    bool first = true;
    
    for (int i = 0; i < 10; ++i) {
        // Extraire le i-ème logit
        auto ct_logit_i = ICiphertext::make();
        
        if (i == 0) {
            *ct_logit_i = logits_enc;
        } else {
            auto it = rot_keys.find(i);
            if (it != rot_keys.end()) {
                eval.rot(logits_enc, -i, *ct_logit_i, *(it->second));
            }
        }
        
        // Extraire le slot 0
        auto ct_logit_slot0 = ICiphertext::make();
        auto ptxt_mask_logit = IPlaintext::make();
        eval.levelDownTo(*ptxt_mask, *ptxt_mask_logit, eval.getLevel(*ct_logit_i));
        eval.mul(*ct_logit_i, *ptxt_mask_logit, *ct_logit_slot0);
        eval.rescale(*ct_logit_slot0, *ct_logit_slot0);
        
        // Comparer avec le max
        auto ct_eq = homomorphic_gt(*ct_max_slot0, *ct_logit_slot0, eval, relin_key);
        
        // Rotation à la position i
        if (i > 0) {
            auto ct_rot = ICiphertext::make();
            auto it = rot_keys.find(i);
            if (it != rot_keys.end()) {
                eval.rot(*ct_eq, i, *ct_rot, *(it->second));
                ct_eq = std::move(ct_rot);
            }
        }
        
        // Accumuler
        if (first) {
            *ct_onehot = *ct_eq;
            first = false;
        } else {
            auto ct_add = ICiphertext::make();
            eval.add(*ct_onehot, *ct_eq, *ct_add);
            *ct_onehot = *ct_add;
        }
    }
    
    std::cout << "    ✅ One-hot vector généré" << std::endl;
    
    return ct_onehot;
}

} // namespace fhe_cnn