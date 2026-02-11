#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <cmath>

namespace fhe_cnn {

using namespace heaan;

double compute_scale_factor(const std::vector<double>& activations) {
    double max_val = 0.0;
    for (double v : activations) {
        max_val = std::max(max_val, std::abs(v));
    }
    return max_val + 0.1;  // Un peu de marge
}

Ptr<ICiphertext> scale_ciphertext(
    const ICiphertext& ctxt,
    double factor,
    HomEval& eval
) {
    auto ct_scaled = ICiphertext::make();
    eval.mul(ctxt, factor, *ct_scaled);
    eval.rescale(*ct_scaled, *ct_scaled);
    return ct_scaled;
}

} // namespace fhe_cnn