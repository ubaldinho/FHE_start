#include "fhe_cnn/conv2d.hpp"
#include <iostream>

namespace fhe_cnn {

using namespace heaan;

Ptr<ICiphertext> homomorphic_conv2d(
    const ICiphertext& input_enc,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    int in_c,
    int in_h,
    int in_w,
    int out_c,
    int kernel,
    int out_h,
    int out_w,
    const ISecretKey& sk,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval
) {
    std::cout << "🔷 Conv2D: " << in_c << "×" << in_h << "×" << in_w 
              << " → " << out_c << "×" << out_h << "×" << out_w << std::endl;
    
    // TODO: Implémentation complète
    auto ct_result = ICiphertext::make();
    return ct_result;
}

} // namespace fhe_cnn