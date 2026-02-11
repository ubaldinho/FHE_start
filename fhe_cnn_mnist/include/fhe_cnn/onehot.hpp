#ifndef FHE_CNN_ONEHOT_HPP
#define FHE_CNN_ONEHOT_HPP

#include <HEAAN2/HEAAN2.hpp>
#include <map>

namespace fhe_cnn {

/**
 * Comparaison homomorphe (approximation polynomiale)
 * 
 * @param x_enc Premier ciphertext
 * @param y_enc Second ciphertext
 * @param eval Évaluateur homomorphe
 * @param relin_key Clé de relinéarisation
 * @return Ciphertext avec 1 si x > y, 0 sinon (approximé)
 */
heaan::Ptr<heaan::ICiphertext> homomorphic_gt(
    const heaan::ICiphertext& x_enc,
    const heaan::ICiphertext& y_enc,
    heaan::HomEval& eval,
    const heaan::ISwKey& relin_key
);

/**
 * Trouver le maximum parmi les 10 premiers slots
 * 
 * @param logits_enc Ciphertext avec 10 logits dans slots 0-9
 * @param rot_keys Clés de rotation
 * @param eval Évaluateur homomorphe
 * @param relin_key Clé de relinéarisation
 * @return Ciphertext avec la valeur max dans tous les slots
 */
heaan::Ptr<heaan::ICiphertext> homomorphic_max(
    const heaan::ICiphertext& logits_enc,
    std::map<int, heaan::Ptr<heaan::ISwKey>>& rot_keys,
    heaan::HomEval& eval,
    const heaan::ISwKey& relin_key
);

/**
 * Convertir les logits en one-hot vector
 * 
 * @param logits_enc Ciphertext avec 10 logits dans slots 0-9
 * @param sk Clé secrète (pour bootstrap si nécessaire)
 * @param rot_keys Clés de rotation
 * @param eval Évaluateur homomorphe
 * @param relin_key Clé de relinéarisation
 * @return Ciphertext one-hot vector (1 au max, 0 ailleurs)
 */
heaan::Ptr<heaan::ICiphertext> homomorphic_onehot(
    const heaan::ICiphertext& logits_enc,
    const heaan::ISecretKey& sk,
    std::map<int, heaan::Ptr<heaan::ISwKey>>& rot_keys,
    heaan::HomEval& eval,
    const heaan::ISwKey& relin_key
);

} // namespace fhe_cnn

#endif // FHE_CNN_ONEHOT_HPP