#ifndef FHE_CNN_BOOTSTRAPPING_HPP
#define FHE_CNN_BOOTSTRAPPING_HPP

#include <HEAAN2/HEAAN2.hpp>

namespace fhe_cnn {

/**
 * Bootstrapping pour rafraîchir les niveaux d'un ciphertext
 * 
 * @param ctxt Ciphertext à rafraîchir (sera modifié)
 * @param sk Clé secrète
 * @param eval Évaluateur homomorphe
 * @param preset_id Paramètres (doit être compatible avec bootstrapping)
 */
void bootstrap_ciphertext(
    heaan::Ptr<heaan::ICiphertext>& ctxt,
    const heaan::ISecretKey& sk,
    heaan::HomEval& eval,
    heaan::PresetParamsId preset_id = heaan::PresetParamsId::F16Opt_Gr
);

/**
 * Vérifier si le bootstrapping est nécessaire
 * 
 * @param ctxt Ciphertext à vérifier
 * @param threshold Niveau minimum avant bootstrap
 * @return true si besoin de bootstrap
 */
bool need_bootstrap(
    const heaan::ICiphertext& ctxt,
    heaan::HomEval& eval,
    int threshold = 3
);

} // namespace fhe_cnn

#endif // FHE_CNN_BOOTSTRAPPING_HPP