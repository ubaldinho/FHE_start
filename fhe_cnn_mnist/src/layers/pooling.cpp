#include "fhe_cnn/pooling.hpp"
#include <iostream>

namespace fhe_cnn {

using namespace heaan;

Ptr<ICiphertext> homomorphic_avgpool2d(
    const ICiphertext& input_enc,
    int c,
    int h,
    int w,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval
) {
    std::cout << "🔷 AvgPool2d: " << c << "×" << h << "×" << w 
              << " → " << c << "×" << h/2 << "×" << w/2 << std::endl;
    
    int out_h = h / 2;
    int out_w = w / 2;
    
    // ------------------------------------------------------------
    // 1. Créer une copie du ciphertext d'entrée
    // ------------------------------------------------------------
    auto ct_sum = ICiphertext::make();
    *ct_sum = input_enc;  // Copie
    
    // ------------------------------------------------------------
    // 2. Additionner le pixel à droite (shift = 1)
    // ------------------------------------------------------------
    auto it1 = rot_keys.find(1);
    if (it1 != rot_keys.end()) {
        auto ct_rot1 = ICiphertext::make();
        eval.rot(*ct_sum, 1, *ct_rot1, *(it1->second));
        
        auto ct_add1 = ICiphertext::make();
        eval.add(*ct_sum, *ct_rot1, *ct_add1);
        ct_sum = std::move(ct_add1);
    }
    
    // ------------------------------------------------------------
    // 3. Additionner le pixel en bas (shift = w)
    // ------------------------------------------------------------
    auto itw = rot_keys.find(w);
    if (itw != rot_keys.end()) {
        auto ct_rotw = ICiphertext::make();
        eval.rot(*ct_sum, w, *ct_rotw, *(itw->second));
        
        auto ct_addw = ICiphertext::make();
        eval.add(*ct_sum, *ct_rotw, *ct_addw);
        ct_sum = std::move(ct_addw);
    }
    
    // ------------------------------------------------------------
    // 4. Additionner le pixel en bas à droite (shift = w + 1)
    // ------------------------------------------------------------
    int shift_w1 = w + 1;
    auto itw1 = rot_keys.find(shift_w1);
    if (itw1 != rot_keys.end()) {
        auto ct_rotw1 = ICiphertext::make();
        eval.rot(*ct_sum, shift_w1, *ct_rotw1, *(itw1->second));
        
        auto ct_addw1 = ICiphertext::make();
        eval.add(*ct_sum, *ct_rotw1, *ct_addw1);
        ct_sum = std::move(ct_addw1);
    }
    
    // ------------------------------------------------------------
    // 5. Multiplier par 0.25 pour la moyenne
    // ------------------------------------------------------------
    auto ct_result = ICiphertext::make();
    eval.mul(*ct_sum, 0.25, *ct_result);
    eval.rescale(*ct_result, *ct_result);
    
    // ------------------------------------------------------------
    // 6. Nettoyer les slots: on ne garde qu'un pixel sur 2
    //    (les autres sont des déchets)
    // ------------------------------------------------------------
    // Pour l'instant on garde tel quel, on fera le masking plus tard
    
    std::cout << "    ✅ AvgPool2d terminé, niveau: " 
              << eval.getLevel(*ct_result) << std::endl;
    
    return ct_result;
}

} // namespace fhe_cnn