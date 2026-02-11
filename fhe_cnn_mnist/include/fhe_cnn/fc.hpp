#ifndef FHE_CNN_FC_HPP
#define FHE_CNN_FC_HPP

#include <HEAAN2/HEAAN2.hpp>
#include <vector>
#include <map>

namespace fhe_cnn {

/**
 * Fully Connected layer homomorphique
 * 
 * @param x_enc Ciphertext d'entrée (vecteur packé dans les slots)
 * @param weight Poids [out_features * in_features]
 * @param bias Bias [out_features]
 * @param in_features Taille d'entrée
 * @param out_features Taille de sortie
 * @param sk Clé secrète
 * @param rot_keys Clés de rotation pour BSGS
 * @param eval Évaluateur homomorphe
 */
heaan::Ptr<heaan::ICiphertext> homomorphic_fc(
    const heaan::ICiphertext& x_enc,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    int in_features,
    int out_features,
    const heaan::ISecretKey& sk,
    std::map<int, heaan::Ptr<heaan::ISwKey>>& rot_keys,
    heaan::HomEval& eval
);

} // namespace fhe_cnn

#endif // FHE_CNN_FC_HPP