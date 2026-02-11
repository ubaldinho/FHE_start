#ifndef FHE_CNN_POOLING_HPP
#define FHE_CNN_POOLING_HPP

#include <HEAAN2/HEAAN2.hpp>

#include <map>

namespace fhe_cnn {

/**
 * Average Pooling 2x2 homomorphique
 * 
 * Stratégie: 
 * 1. Additionner les 4 pixels du pool par rotations
 * 2. Multiplier par 0.25 pour la moyenne
 * 
 * @param input_enc Ciphertext d'entrée (packé [c][h][w])
 * @param c Nombre de canaux
 * @param h Hauteur d'entrée
 * @param w Largeur d'entrée
 * @param rot_keys Clés de rotation (pour shift=1 et shift=w)
 * @param eval Évaluateur homomorphe
 * @return Ciphertext après pooling (c × h/2 × w/2)
 */
heaan::Ptr<heaan::ICiphertext> homomorphic_avgpool2d(
    const heaan::ICiphertext& input_enc,
    int c,
    int h,
    int w,
    std::map<int, heaan::Ptr<heaan::ISwKey>>& rot_keys,
    heaan::HomEval& eval
);

} // namespace fhe_cnn

#endif // FHE_CNN_POOLING_HPP