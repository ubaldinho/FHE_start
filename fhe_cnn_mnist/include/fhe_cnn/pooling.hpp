#ifndef FHE_CNN_POOLING_HPP
#define FHE_CNN_POOLING_HPP

#include <HEAAN2/HEAAN2.hpp>

namespace fhe_cnn {

/**
 * Average Pooling 2x2 homomorphique
 * 
 * @param input_enc Ciphertext d'entrée
 * @param c Nombre de canaux
 * @param h Hauteur d'entrée
 * @param w Largeur d'entrée
 * @param eval Évaluateur homomorphe
 * @return Ciphertext après pooling (h/2 × w/2)
 */
heaan::Ptr<heaan::ICiphertext> homomorphic_avgpool2d(
    const heaan::ICiphertext& input_enc,
    int c,
    int h,
    int w,
    heaan::HomEval& eval
);

} // namespace fhe_cnn

#endif // FHE_CNN_POOLING_HPP