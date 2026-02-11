#include "fhe_cnn/conv2d.hpp"
#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <chrono>
#include <random>

using namespace heaan;
using namespace fhe_cnn;

int main() {
    std::cout << "\n🧪 Test Conv2D Layer" << std::endl;
    std::cout << "====================" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // ------------------------------------------------------------
    // 1. Initialisation HEAAN2
    // ------------------------------------------------------------
    std::cout << "\n1. Initialisation HEAAN2..." << std::endl;
    
    auto preset_id = PresetParamsId::F16Opt_Gr;
    
    SKGenerator skgen(preset_id);
    auto sk = skgen.genKey();
    sk->to(Device::CPU);
    
    SwKeyGenerator swkgen(preset_id);
    auto relin_key = swkgen.genRelinKey(*sk);
    
    HomEval eval(preset_id);
    EnDecoder encoder(preset_id);
    EnDecryptor encryptor(preset_id);
    
    // ------------------------------------------------------------
    // 2. Génération des clés de rotation
    // ------------------------------------------------------------
    std::cout << "\n2. Génération des clés de rotation..." << std::endl;
    
    std::map<int, Ptr<ISwKey>> rot_keys;
    int max_rot = 900;  // Pour image 28×28 + décalages
    generate_all_rot_keys(*sk, max_rot, rot_keys);
    
    // ------------------------------------------------------------
    // 3. Création des données de test (petite image)
    // ------------------------------------------------------------
    std::cout << "\n3. Création des données de test..." << std::endl;
    
    int in_c = 1, in_h = 8, in_w = 8;  // 8×8 pour test rapide
    int out_c = 2, kernel = 3;         // Kernel 3×3
    int out_h = in_h - kernel + 1;
    int out_w = in_w - kernel + 1;
    
    // Image aléatoire
    std::vector<double> input(in_c * in_h * in_w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (int i = 0; i < (int)input.size(); ++i) {
        input[i] = dis(gen);
    }
    
    // Poids aléatoires
    std::vector<double> weight(out_c * in_c * kernel * kernel);
    for (int i = 0; i < (int)weight.size(); ++i) {
        weight[i] = dis(gen);
    }
    
    // Bias aléatoires
    std::vector<double> bias(out_c);
    for (int i = 0; i < out_c; ++i) {
        bias[i] = dis(gen);
    }
    
    // ------------------------------------------------------------
    // 4. Chiffrement de l'image
    // ------------------------------------------------------------
    std::cout << "\n4. Chiffrement de l'image..." << std::endl;
    
    auto ct_input = encrypt_image(input, *sk, encoder, encryptor);
    
    // ------------------------------------------------------------
    // 5. Convolution homomorphe
    // ------------------------------------------------------------
    std::cout << "\n5. Exécution Conv2D homomorphe..." << std::endl;
    
    auto ct_output = homomorphic_conv2d(
        *ct_input, weight, bias,
        in_c, in_h, in_w,
        out_c, kernel, out_h, out_w,
        *sk, rot_keys, *relin_key, eval
    );
    
    // ------------------------------------------------------------
    // 6. Déchiffrement
    // ------------------------------------------------------------
    std::cout << "\n6. Déchiffrement..." << std::endl;
    
    auto output = decrypt_result(*ct_output, *sk, encoder, encryptor, out_c * out_h * out_w);
    
    // ------------------------------------------------------------
    // 7. Calcul en clair pour vérification
    // ------------------------------------------------------------
    std::cout << "\n7. Vérification..." << std::endl;
    
    // Reshape pour faciliter le calcul
    std::vector<std::vector<std::vector<double>>> x(
        in_c, std::vector<std::vector<double>>(in_h, std::vector<double>(in_w)));
    
    for (int ic = 0; ic < in_c; ++ic) {
        for (int ih = 0; ih < in_h; ++ih) {
            for (int iw = 0; iw < in_w; ++iw) {
                x[ic][ih][iw] = input[(ic * in_h + ih) * in_w + iw];
            }
        }
    }
    
    std::vector<double> y_clear(out_c * out_h * out_w, 0.0);
    
    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                double sum = bias[oc];
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < kernel; ++kh) {
                        for (int kw = 0; kw < kernel; ++kw) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            int w_idx = (((oc * in_c + ic) * kernel + kh) * kernel + kw);
                            sum += x[ic][ih][iw] * weight[w_idx];
                        }
                    }
                }
                y_clear[(oc * out_h + oh) * out_w + ow] = sum;
            }
        }
    }
    
    // ------------------------------------------------------------
    // 8. Comparaison
    // ------------------------------------------------------------
    std::cout << "\n=== Résultats ===" << std::endl;
    
    double max_err = 0.0;
    int n = out_c * out_h * out_w;
    
    for (int i = 0; i < std::min(5, n); ++i) {
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