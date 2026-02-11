#ifndef FHE_CNN_RELU_HPP
#define FHE_CNN_RELU_HPP

#include <HEAAN2/HEAAN2.hpp>

namespace fhe_cnn {

/**
 * Approximation polynomiale de ReLU (degré 3,5,7...)
 * 
 * @param input_enc Ciphertext d'entrée
 * @param degree Degré du polynôme d'approximation (3, 5 ou 7)
 * @param scale_factor Facteur de scaling avant ReLU (pour être dans [-1,1])
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