#ifndef FHE_CNN_BOOTSTRAPPING_HPP
#define FHE_CNN_BOOTSTRAPPING_HPP

#include <HEAAN2/HEAAN2.hpp>

namespace fhe_cnn {

/**
 * Bootstrapping pour rafraîchir les niveaux
 * 
 * @param ctxt Ciphertext à rafraîchir
 * @param sk Clé secrète
 * @param eval Évaluateur homomorphe
 */
void bootstrap_ciphertext(
    heaan::Ptr<heaan::ICiphertext>& ctxt,
    const heaan::ISecretKey& sk,
    heaan::HomEval& eval
);

} // namespace fhe_cnn

#endif // FHE_CNN_BOOTSTRAPPING_HPP