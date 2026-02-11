#include "fhe_cnn/relu.hpp"
#include <iostream>

namespace fhe_cnn {

using namespace heaan;

Ptr<ICiphertext> homomorphic_relu(
    const ICiphertext& input_enc,
    int degree,
    double scale_factor,
    HomEval& eval,
    const ISwitchingKey& relin_key
) {
    std::cout << "🔷 ReLU (degré " << degree << ")" << std::endl;
    
    // TODO: Implémentation polynomiale
    auto ct_result = ICiphertext::make();
    *ct_result = input_enc;  // Copie temporaire
    return ct_result;
}

} // namespace fhe_cnn