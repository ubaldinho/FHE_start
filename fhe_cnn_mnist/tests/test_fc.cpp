#include "fhe_cnn/fc.hpp"
#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <chrono>

using namespace heaan;
using namespace fhe_cnn;

int main() {
    std::cout << "\n🧪 Test FC Layer" << std::endl;
    std::cout << "================" << std::endl;
    
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
    int max_rot = 16;  // Pour test
    generate_all_rot_keys(*sk, max_rot, rot_keys);
    
    // ------------------------------------------------------------
    // 3. Création des données de test
    // ------------------------------------------------------------
    std::cout << "\n3. Création des données de test..." << std::endl;
    
    int in_features = 8;
    int out_features = 4;
    
    // Vecteur d'entrée [1,2,3,4,5,6,7,8]
    std::vector<double> x(in_features);
    for (int i = 0; i < in_features; ++i) x[i] = i + 1;
    
    // Poids: matrice 4×8 avec valeurs aléatoires
    std::vector<double> weight(out_features * in_features);
    for (int i = 0; i < out_features * in_features; ++i) {
        weight[i] = (double)rand() / RAND_MAX;
    }
    
    // Bias: [0.1, 0.2, 0.3, 0.4]
    std::vector<double> bias(out_features);
    for (int i = 0; i < out_features; ++i) bias[i] = (i + 1) * 0.1;
    
    // ------------------------------------------------------------
    // 4. Chiffrement du vecteur d'entrée
    // ------------------------------------------------------------
    std::cout << "\n4. Chiffrement du vecteur d'entrée..." << std::endl;
    
    int log_slots = sk->logDegree() - 1;
    Message<Complex> msg_x(log_slots, Device::CPU);
    
    for (int i = 0; i < in_features; ++i) msg_x[i] = Complex(x[i], 0.0);
    for (int i = in_features; i < (1 << log_slots); ++i) msg_x[i] = Complex(0.0, 0.0);
    
    auto ptxt_x = IPlaintext::make();
    encoder.encode(msg_x, *ptxt_x);
    
    auto ct_x = ICiphertext::make();
    encryptor.encrypt(*ptxt_x, *sk, *ct_x);
    
    std::cout << "    Niveau: " << eval.getLevel(*ct_x) << std::endl;
    
    // ------------------------------------------------------------
    // 5. FC homomorphe
    // ------------------------------------------------------------
    std::cout << "\n5. Exécution FC homomorphe..." << std::endl;
    
    auto ct_y = homomorphic_fc(
        *ct_x, weight, bias, in_features, out_features, *sk, rot_keys, eval
    );
    
    // ------------------------------------------------------------
    // 6. Déchiffrement
    // ------------------------------------------------------------
    std::cout << "\n6. Déchiffrement..." << std::endl;
    
    auto ptxt_y = IPlaintext::make();
    encryptor.decrypt(*ct_y, *sk, *ptxt_y);
    
    Message<Complex> msg_y;
    encoder.decode(*ptxt_y, msg_y);
    msg_y.to(Device::CPU);
    
    // ------------------------------------------------------------
    // 7. Calcul en clair
    // ------------------------------------------------------------
    std::cout << "\n7. Vérification..." << std::endl;
    
    std::vector<double> y_clear(out_features, 0.0);
    for (int i = 0; i < out_features; ++i) {
        y_clear[i] = bias[i];
        for (int j = 0; j < in_features; ++j) {
            y_clear[i] += weight[i * in_features + j] * x[j];
        }
    }
    
    // ------------------------------------------------------------
    // 8. Comparaison
    // ------------------------------------------------------------
    std::cout << "\n=== Résultats ===" << std::endl;
    
    double max_err = 0.0;
    int n = out_features;
    
    for (int i = 0; i < out_features; ++i) {
        double y_fhe = msg_y[i].real() / n;  // Division par n !
        double err = std::abs(y_fhe - y_clear[i]);
        max_err = std::max(max_err, err);
        
        std::cout << "  [" << i << "] Clair: " << y_clear[i] 
                  << ", FHE: " << y_fhe
                  << ", Erreur: " << err << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n=== Statistiques ===" << std::endl;
    std::cout << "  Erreur max: " << max_err << std::endl;
    std::cout << "  Erreur (log2): " << std::log2(max_err) << " bits" << std::endl;
    std::cout << "  Temps: " << duration.count() << " ms" << std::endl;
    
    if (max_err < 1e-6) {
        std::cout << "\n✅ TEST PASSÉ!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ TEST ÉCHOUÉ!" << std::endl;
        return 1;
    }
}