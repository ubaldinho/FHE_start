#include "fhe_cnn/relu.hpp"
#include <iostream>
#include <cmath>

namespace fhe_cnn {

using namespace heaan;

Ptr<ICiphertext> homomorphic_relu(
    const ICiphertext& input_enc,
    int degree,
    double scale_factor,
    HomEval& eval,
    const ISwKey& relin_key
) {
    std::cout << "🔷 ReLU (degré " << degree << ", scale=" << scale_factor << ")" << std::endl;
    
    // ------------------------------------------------------------
    // 1. Mettre à l'échelle dans [-1, 1]
    // ------------------------------------------------------------
    auto ct_scaled = ICiphertext::make();
    eval.mul(input_enc, 1.0 / scale_factor, *ct_scaled);
    eval.rescale(*ct_scaled, *ct_scaled);
    
    // ------------------------------------------------------------
    // 2. Évaluation polynomiale selon le degré
    // ------------------------------------------------------------
    Ptr<ICiphertext> ct_result;
    
    if (degree == 3) {
        // ReLU ≈ 0.2978 + 0.5x + 0.2978x³
        std::cout << "    Polynôme degré 3: 0.2978 + 0.5x + 0.2978x³" << std::endl;
        
        // x²
        auto ct_x2 = ICiphertext::make();
        eval.tensor(*ct_scaled, *ct_scaled, *ct_x2);
        eval.relin(*ct_x2, relin_key);
        eval.rescale(*ct_x2, *ct_x2);
        
        // x³ = x² * x
        auto ct_x3 = ICiphertext::make();
        eval.tensor(*ct_x2, *ct_scaled, *ct_x3);
        eval.relin(*ct_x3, relin_key);
        eval.rescale(*ct_x3, *ct_x3);
        
        // 0.2978 * x³
        auto ct_term3 = ICiphertext::make();
        eval.mul(*ct_x3, 0.2978, *ct_term3);
        eval.rescale(*ct_term3, *ct_term3);
        
        // 0.5 * x
        auto ct_term1 = ICiphertext::make();
        eval.mul(*ct_scaled, 0.5, *ct_term1);
        eval.rescale(*ct_term1, *ct_term1);
        
        // 0.2978 + 0.5x + 0.2978x³
        auto ct_temp = ICiphertext::make();
        eval.add(*ct_term1, *ct_term3, *ct_temp);
        
        ct_result = ICiphertext::make();
        eval.add(*ct_temp, 0.2978, *ct_result);
        
    } else if (degree == 5) {
        // ReLU ≈ 0.125 + 0.5x + 0.375x² + 0.125x³ + 0.0625x⁴ + 0.0625x⁵
        std::cout << "    Polynôme degré 5" << std::endl;
        
        // x²
        auto ct_x2 = ICiphertext::make();
        eval.tensor(*ct_scaled, *ct_scaled, *ct_x2);
        eval.relin(*ct_x2, relin_key);
        eval.rescale(*ct_x2, *ct_x2);
        
        // x³ = x² * x
        auto ct_x3 = ICiphertext::make();
        eval.tensor(*ct_x2, *ct_scaled, *ct_x3);
        eval.relin(*ct_x3, relin_key);
        eval.rescale(*ct_x3, *ct_x3);
        
        // x⁴ = x² * x²
        auto ct_x4 = ICiphertext::make();
        eval.tensor(*ct_x2, *ct_x2, *ct_x4);
        eval.relin(*ct_x4, relin_key);
        eval.rescale(*ct_x4, *ct_x4);
        
        // x⁵ = x² * x³
        auto ct_x5 = ICiphertext::make();
        eval.tensor(*ct_x2, *ct_x3, *ct_x5);
        eval.relin(*ct_x5, relin_key);
        eval.rescale(*ct_x5, *ct_x5);
        
        // 0.5x
        auto ct_c1 = ICiphertext::make();
        eval.mul(*ct_scaled, 0.5, *ct_c1);
        eval.rescale(*ct_c1, *ct_c1);
        
        // 0.375x²
        auto ct_c2 = ICiphertext::make();
        eval.mul(*ct_x2, 0.375, *ct_c2);
        eval.rescale(*ct_c2, *ct_c2);
        
        // 0.125x³
        auto ct_c3 = ICiphertext::make();
        eval.mul(*ct_x3, 0.125, *ct_c3);
        eval.rescale(*ct_c3, *ct_c3);
        
        // 0.0625x⁴
        auto ct_c4 = ICiphertext::make();
        eval.mul(*ct_x4, 0.0625, *ct_c4);
        eval.rescale(*ct_c4, *ct_c4);
        
        // 0.0625x⁵
        auto ct_c5 = ICiphertext::make();
        eval.mul(*ct_x5, 0.0625, *ct_c5);
        eval.rescale(*ct_c5, *ct_c5);
        
        // Somme
        auto ct_sum1 = ICiphertext::make();
        eval.add(*ct_c1, *ct_c2, *ct_sum1);
        
        auto ct_sum2 = ICiphertext::make();
        eval.add(*ct_sum1, *ct_c3, *ct_sum2);
        
        auto ct_sum3 = ICiphertext::make();
        eval.add(*ct_sum2, *ct_c4, *ct_sum3);
        
        auto ct_sum4 = ICiphertext::make();
        eval.add(*ct_sum3, *ct_c5, *ct_sum4);
        
        ct_result = ICiphertext::make();
        eval.add(*ct_sum4, 0.125, *ct_result);
        
    } else {
        // Degré 7 par défaut (approximation plus précise)
        std::cout << "    Polynôme degré 7" << std::endl;
        
        // Coefficients pour degré 7 (approximation minimax)
        double c0 = 0.0542;
        double c1 = 0.5;
        double c2 = 0.3021;
        double c3 = 0.0859;
        double c4 = 0.0352;
        double c5 = 0.0137;
        double c6 = 0.0049;
        double c7 = 0.0019;
        
        // Calculer les puissances avec Horner
        auto ct_pow = ICiphertext::make();
        *ct_pow = *ct_scaled;
        
        // TODO: Implémentation complète degré 7
        ct_result = ICiphertext::make();
        *ct_result = *ct_scaled;  // Temporaire
    }
    
    // ------------------------------------------------------------
    // 3. Remettre à l'échelle originale
    // ------------------------------------------------------------
    auto ct_restored = ICiphertext::make();
    eval.mul(*ct_result, scale_factor, *ct_restored);
    eval.rescale(*ct_restored, *ct_restored);
    
    std::cout << "    ✅ ReLU terminé, niveau: " 
              << eval.getLevel(*ct_restored) << std::endl;
    
    return ct_restored;
}

} // namespace fhe_cnn