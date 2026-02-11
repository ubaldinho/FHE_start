#include "fhe_cnn/bootstrapping.hpp"
#include <iostream>

namespace fhe_cnn {

using namespace heaan;

void bootstrap_ciphertext(
    Ptr<ICiphertext>& ctxt,
    const ISecretKey& sk,
    HomEval& eval
) {
    std::cout << "🔷 Bootstrapping..." << std::endl;
    
    // TODO: Implémentation bootstrap
    // Nécessite BootKeyPtrs et Bootstrapper
}

} // namespace fhe_cnn