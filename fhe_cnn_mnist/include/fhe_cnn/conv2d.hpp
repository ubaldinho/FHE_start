#ifndef FHE_CNN_CONV2D_HPP
#define FHE_CNN_CONV2D_HPP

#include <HEAAN2/HEAAN2.hpp>
#include <vector>
#include <map>

namespace fhe_cnn {

/**
 * Convolution 2D homomorphique
 * 
 * @param input_enc Ciphertext de l'image d'entrée (packée ligne par ligne)
 * @param weight Poids [out_c][in_c][5][5] en clair
 * @param bias Bias [out_c] en clair
 * @param in_c Canaux d'entrée
 * @param in_h Hauteur d'entrée
 * @param in_w Largeur d'entrée
 * @param out_c Canaux de sortie
 * @param kernel Taille du noyau (5)
 * @param out_h Hauteur de sortie
 * @param out_w Largeur de sortie
 * @param sk Clé secrète
 * @param rot_keys Map des clés de rotation
 * @param eval Évaluateur homomorphe
 * @return Ciphertext de la sortie convoluée
 */
heaan::Ptr<heaan::ICiphertext> homomorphic_conv2d(
    const heaan::ICiphertext& input_enc,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    int in_c,
    int in_h,
    int in_w,
    int out_c,
    int kernel,
    int out_h,
    int out_w,
    const heaan::ISecretKey& sk,
    std::map<int, heaan::Ptr<heaan::ISwKey>>& rot_keys,
    heaan::HomEval& eval
);

} // namespace fhe_cnn

#endif // FHE_CNN_CONV2D_HPP