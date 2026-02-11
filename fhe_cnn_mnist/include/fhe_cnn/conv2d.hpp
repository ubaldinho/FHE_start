#ifndef FHE_CNN_CONV2D_HPP
#define FHE_CNN_CONV2D_HPP

#include <HEAAN2/HEAAN2.hpp>
#include <vector>
#include <map>

namespace fhe_cnn {

/**
 * Convolution 2D homomorphique - Version optimisée
 * 
 * Stratégie: 
 * 1. Packer l'image d'entrée ligne par ligne dans les slots
 * 2. Pour chaque position du kernel, extraire les 25 pixels
 * 3. Diagonal method pour calculer toutes les convolutions en parallèle
 * 
 * @param input_enc Ciphertext avec image packée (H×W slots)
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
 * @param relin_key Clé de relinéarisation
 * @param eval Évaluateur homomorphe
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
    const heaan::ISwKey& relin_key,
    heaan::HomEval& eval
);

} // namespace fhe_cnn

#endif // FHE_CNN_CONV2D_HPP