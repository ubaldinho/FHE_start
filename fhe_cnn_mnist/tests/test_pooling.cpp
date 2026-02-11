#include "fhe_cnn/pooling.hpp"
#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <chrono>
#include <random>

using namespace heaan;
using namespace fhe_cnn;

int main() {
    std::cout << "\n🧪 Test AveragePool Layer" << std::endl;
    std::cout << "=========================" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // ------------------------------------------------------------
    // 1. Initialisation HEAAN2
    // ------------------------------------------------------------
    std::cout << "\n1. Initialisation HEAAN2..." << std::endl;
    
    auto preset_id = PresetParamsId::F16Opt_Gr;
    
    SKGenerator skgen(preset_id);
    auto sk = skgen.genKey();
    sk->to(Device::CPU);
    
    HomEval eval(preset_id);
    EnDecoder encoder(preset_id);
    EnDecryptor encryptor(preset_id);
    
    // ------------------------------------------------------------
    // 2. Génération des clés de rotation
    // ------------------------------------------------------------
    std::cout << "\n2. Génération des clés de rotation..." << std::endl;
    
    std::map<int, Ptr<ISwKey>> rot_keys;
    SwKeyGenerator swkgen(preset_id);
    
    // Clés nécessaires pour pooling
    for (int rot : {1, 4, 5}) {  // w=4 pour test
        auto rot_key = swkgen.genRotKey(*sk, rot);
        rot_keys[rot] = std::move(rot_key);
    }
    
    // ------------------------------------------------------------
    // 3. Création des données de test
    // ------------------------------------------------------------
    std::cout << "\n3. Création des données de test..." << std::endl;
    
    int c = 2, h = 4, w = 4;
    int out_h = h / 2, out_w = w / 2;
    
    std::vector<double> input(c * h * w);
    for (int i = 0; i < (int)input.size(); ++i) {
        input[i] = i + 1;  // 1,2,3,... pour test
    }
    
    // ------------------------------------------------------------
    // 4. Chiffrement
    // ------------------------------------------------------------
    std::cout << "\n4. Chiffrement..." << std::endl;
    
    auto ct_input = encrypt_image(input, *sk, encoder, encryptor);
    
    // ------------------------------------------------------------
    // 5. AveragePool homomorphe
    // ------------------------------------------------------------
    std::cout << "\n5. Exécution AveragePool..." << std::endl;
    
    auto ct_output = homomorphic_avgpool2d(*ct_input, c, h, w, rot_keys, eval);
    
    // ------------------------------------------------------------
    // 6. Déchiffrement
    // ------------------------------------------------------------
    std::cout << "\n6. Déchiffrement..." << std::endl;
    
    auto output = decrypt_result(*ct_output, *sk, encoder, encryptor, c * out_h * out_w);
    
    // ------------------------------------------------------------
    // 7. Calcul en clair
    // ------------------------------------------------------------
    std::cout << "\n7. Vérification..." << std::endl;
    
    std::vector<double> y_clear(c * out_h * out_w, 0.0);
    
    for (int ch = 0; ch < c; ++ch) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                double sum = 0;
                for (int kh = 0; kh < 2; ++kh) {
                    for (int kw = 0; kw < 2; ++kw) {
                        int ih = oh * 2 + kh;
                        int iw = ow * 2 + kw;
                        int idx = (ch * h + ih) * w + iw;
                        sum += input[idx];
                    }
                }
                y_clear[(ch * out_h + oh) * out_w + ow] = sum / 4.0;
            }
        }
    }
    
    // ------------------------------------------------------------
    // 8. Comparaison
    // ------------------------------------------------------------
    std::cout << "\n=== Résultats ===" << std::endl;
    
    double max_err = 0.0;
    for (int i = 0; i < std::min(5, (int)output.size()); ++i) {
        double err = std::abs(output[i] - y_clear[i]);
        max_err = std::max(max_err, err);
        std::cout << "  [" << i << "] Clair: " << y_clear[i] 
                  << ", FHE: " << output[i]
                  << ", Erreur: " << err << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n=== Statistiques ===" << std::endl;
    std::cout << "  Erreur max: " << max_err << std::endl;
    std::cout << "  Erreur (log2): " << std::log2(max_err) << " bits" << std::endl;
    std::cout << "  Temps: " << duration.count() << " ms" << std::endl;
    
    if (max_err < 1e-5) {
        std::cout << "\n✅ TEST PASSÉ!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ TEST ÉCHOUÉ!" << std::endl;
        return 1;
    }
}