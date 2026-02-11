#include "fhe_cnn/utils.hpp"
#include <iostream>

namespace fhe_cnn {

using namespace heaan;

void generate_all_rot_keys(
    const ISecretKey& sk,
    int max_rot,
    std::map<int, Ptr<ISwKey>>& rot_keys
) {
    std::cout << "🔑 Génération des clés de rotation (0.." << max_rot-1 << ")..." << std::endl;
    
    SwKeyGenerator swkgen(PresetParamsId::F16Opt_Gr);
    
    // Puissances de 2 pour rotate-and-sum
    for (int shift = 1; shift < max_rot; shift <<= 1) {
        if (rot_keys.find(shift) == rot_keys.end()) {
            std::cout << "    Rot key " << shift << "..." << std::endl;
            auto rot_key = swkgen.genRotKey(sk, shift);
            rot_keys[shift] = std::move(rot_key);
        }
    }
    
    // Baby steps (1..sqrt(max_rot))
    int n2 = (int)std::sqrt(max_rot);
    for (int rot = 1; rot < n2; ++rot) {
        if (rot_keys.find(rot) == rot_keys.end()) {
            std::cout << "    Baby step key " << rot << "..." << std::endl;
            auto rot_key = swkgen.genRotKey(sk, rot);
            rot_keys[rot] = std::move(rot_key);
        }
    }
    
    // Giant steps (multiples de n2)
    int n1 = max_rot / n2;
    for (int j = 1; j < n1; ++j) {
        int rot = j * n2;
        if (rot_keys.find(rot) == rot_keys.end()) {
            std::cout << "    Giant step key " << rot << "..." << std::endl;
            auto rot_key = swkgen.genRotKey(sk, rot);
            rot_keys[rot] = std::move(rot_key);
        }
    }
    
    // Rotations individuelles pour répartition finale
    for (int rot = 1; rot < max_rot; ++rot) {
        if (rot_keys.find(rot) == rot_keys.end()) {
            std::cout << "    Final key " << rot << "..." << std::endl;
            auto rot_key = swkgen.genRotKey(sk, rot);
            rot_keys[rot] = std::move(rot_key);
        }
    }
    
    std::cout << "  ✅ " << rot_keys.size() << " clés générées" << std::endl;
}

} // namespace fhe_cnn