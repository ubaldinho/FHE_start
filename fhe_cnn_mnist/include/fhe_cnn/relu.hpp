#ifndef FHE_CNN_RELU_HPP
#define FHE_CNN_RELU_HPP

#include <HEAAN2/HEAAN2.hpp>

namespace fhe_cnn {

/**
 * Approximation polynomiale de ReLU (degré 3, 5 ou 7)
 * 
 * Polynomiales d'approximation sur [-1, 1]:
 * - Degré 3: 0.2978 + 0.5x + 0.2978x^3
 * - Degré 5: 0.125 + 0.5x + 0.375x^2 + 0.125x^3 + 0.0625x^4 + 0.0625x^5
 * - Degré 7: Approximation plus précise
 * 
 * @param input_enc Ciphertext d'entrée
 * @param degree Degré du polynôme (3, 5 ou 7)
 * @param scale_factor Facteur de scaling (entrée doit être dans [-scale_factor, scale_factor])
 * @param eval Évaluateur homomorphe
 * @param relin_key Clé de relinéarisation
 * @return Ciphertext après ReLU approximé
 */
heaan::Ptr<heaan::ICiphertext> homomorphic_relu(
    const heaan::ICiphertext& input_enc,
    int degree,
    double scale_factor,
    heaan::HomEval& eval,
    const heaan::ISwKey& relin_key
);

} // namespace fhe_cnn

#endif // FHE_CNN_RELU_HPP