#include "HEAAN2/HEAAN2.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace heaan;

const auto preset_id = PresetParamsId::F16Opt_Gr;
const auto device = Device::CPU;

Ptr<ICiphertext> goldschmidtInverse(
    const ICiphertext& ct_x,
    int d,
    const ISecretKey& sk,
    const ISwKey& relin_key,
    HomEval& eval
) {
    std::cout << "\n=== Goldschmidt: Calcul de 1/x ===" << std::endl;
    std::cout << "  Itérations: " << d << std::endl;
    
    // ------------------------------------------------------------
    // ct₁ = 2 - x  →  eval.sub(ct_x, Complex(2,0), ct1) ? NON!
    // Il faut: constante - ciphertext
    // Mais sub() ne prend que (ciphertext, constante) ou (plaintext, plaintext)
    // Solution: 2 - x = 2 + (-x)  ou mieux: créer plaintext avec 2 partout
    // ------------------------------------------------------------
    
    // Méthode propre: créer un plaintext avec 2.0 dans tous les slots
    Message<Complex> msg_two(sk.logDegree() - 1, device);
    for (int i = 0; i < (1 << msg_two.logSlots()); ++i) {
        msg_two[i] = Complex(2.0, 0.0);
    }
    auto ptxt_two = IPlaintext::make();
    EnDecoder encoder(preset_id);
    encoder.encode(msg_two, *ptxt_two);
    
    // ct₁ = 2 - x = ptxt_two - ct_x
    // Pas de sub direct ciphertext-plaintext ? Il faut faire: ct₁ = (-ct_x) + 2
    auto ct1 = ICiphertext::make();
    eval.neg(ct_x, *ct1);           // ct1 = -x
    eval.add(*ct1, *ptxt_two, *ct1); // ct1 = -x + 2 = 2 - x
    std::cout << "  ct1 = 2 - x, niveau: " << eval.getLevel(*ct1) << std::endl;
    
    // ------------------------------------------------------------
    // ct₂ = 1 - x
    // ------------------------------------------------------------
    Message<Complex> msg_one(sk.logDegree() - 1, device);
    for (int i = 0; i < (1 << msg_one.logSlots()); ++i) {
        msg_one[i] = Complex(1.0, 0.0);
    }
    auto ptxt_one = IPlaintext::make();
    encoder.encode(msg_one, *ptxt_one);
    
    auto ct2 = ICiphertext::make();
    eval.neg(ct_x, *ct2);           // ct2 = -x
    eval.add(*ct2, *ptxt_one, *ct2); // ct2 = -x + 1 = 1 - x
    std::cout << "  ct2 = 1 - x, niveau: " << eval.getLevel(*ct2) << std::endl;
    
    // ------------------------------------------------------------
    // Boucle principale
    // ------------------------------------------------------------
    for (int i = 1; i <= d; ++i) {
        std::cout << "\n  Itération " << i << ":" << std::endl;
        
        // --------------------------------------------------------
        // ct₂ = ct₂²
        // --------------------------------------------------------
        auto ct2_sq = ICiphertext::make();
        eval.tensor(*ct2, *ct2, *ct2_sq);
        eval.relin(*ct2_sq, relin_key);
        eval.rescale(*ct2_sq, *ct2_sq);
        *ct2 = *ct2_sq;
        std::cout << "    ct2 = ct2², niveau: " << eval.getLevel(*ct2) << std::endl;
        
        // --------------------------------------------------------
        // one_plus_ct2 = 1 + ct₂
        // --------------------------------------------------------
        auto one_plus_ct2 = ICiphertext::make();
        eval.add(*ct2, *ptxt_one, *one_plus_ct2);  // ct2 + 1
        std::cout << "    1 + ct2, niveau: " << eval.getLevel(*one_plus_ct2) << std::endl;
        
        // --------------------------------------------------------
        // ct₁ = ct₁ · (1 + ct₂)
        // --------------------------------------------------------
        auto ct1_new = ICiphertext::make();
        
        // Vérifier les niveaux
        if (eval.getLevel(*ct1) != eval.getLevel(*one_plus_ct2)) {
            auto ct1_leveled = ICiphertext::make();
            eval.levelDownTo(*ct1, *ct1_leveled, eval.getLevel(*one_plus_ct2));
            *ct1 = *ct1_leveled;
        }
        
        eval.tensor(*ct1, *one_plus_ct2, *ct1_new);
        eval.relin(*ct1_new, relin_key);
        eval.rescale(*ct1_new, *ct1_new);
        *ct1 = *ct1_new;
        std::cout << "    ct1 = ct1 * (1 + ct2), niveau: " << eval.getLevel(*ct1) << std::endl;
    }
    
    return ct1;
}

int main() {
    std::cout << "=== Goldschmidt: Inverse homomorphe ===" << std::endl;
    
    // 1. Génération des clés
    std::cout << "\n1. Génération des clés..." << std::endl;
    SKGenerator skgen(preset_id);
    auto sk = skgen.genKey();
    sk->to(device);
    
    SwKeyGenerator swkgen(preset_id);
    auto relin_key = swkgen.genRelinKey(*sk);
    
    // 2. Données : x dans [0,2]
    int log_slots = sk->logDegree() - 1;
    int num_slots = 1 << log_slots;
    std::cout << "\n2. Création des données (" << num_slots << " slots)..." << std::endl;
    
    Message<Complex> msg_x(log_slots, device);
    for (int i = 0; i < num_slots; ++i) {
        double val = 0.5 + (1.5 * rand() / RAND_MAX);  // [0.5, 2.0]
        msg_x[i] = Complex(val, 0.0);
    }
    
    // 3. Encoder et chiffrer
    auto ptxt_x = IPlaintext::make();
    EnDecoder encoder(preset_id);
    encoder.encode(msg_x, *ptxt_x);
    
    auto ct_x = ICiphertext::make();
    EnDecryptor encryptor(preset_id);
    encryptor.encrypt(*ptxt_x, *sk, *ct_x);
    std::cout << "  x chiffré, niveau: " << HomEval(preset_id).getLevel(*ct_x) << std::endl;
    
    // 4. Calcul de l'inverse
    int iterations = 5;  // d = 5
    HomEval eval(preset_id);
    
    auto ct_inv = goldschmidtInverse(*ct_x, iterations, *sk, *relin_key, eval);
    
    // 5. Déchiffrement et vérification
    std::cout << "\n3. Déchiffrement..." << std::endl;
    auto ptxt_inv = IPlaintext::make();
    encryptor.decrypt(*ct_inv, *sk, *ptxt_inv);
    
    Message<Complex> msg_inv;
    encoder.decode(*ptxt_inv, msg_inv);
    msg_inv.to(Device::CPU);
    
    // 6. Vérification
    std::cout << "\n=== Résultats (5 premiers slots) ===" << std::endl;
    
    double max_err = 0.0;
    for (int i = 0; i < std::min(5, num_slots); ++i) {
        double x = msg_x[i].real();
        double inv_fhe = msg_inv[i].real();
        double inv_true = 1.0 / x;
        double err = std::abs(inv_fhe - inv_true);
        if (err > max_err) max_err = err;
        
        std::cout << "Slot " << i << ": x=" << x 
                  << ", 1/x=" << inv_true
                  << ", FHE=" << inv_fhe
                  << ", erreur=" << err << std::endl;
    }
    
    std::cout << "\n=== Statistiques ===" << std::endl;
    std::cout << "Erreur maximale: " << max_err << std::endl;
    std::cout << "Erreur (log2): " << std::log2(max_err) << " bits" << std::endl;
    std::cout << "Niveau final: " << eval.getLevel(*ct_inv) << std::endl;
    
    return 0;
}