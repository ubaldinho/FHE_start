#include "fhe_cnn/fc.hpp"
#include <iostream>
#include <cmath>

namespace fhe_cnn {

using namespace heaan;

Ptr<ICiphertext> homomorphic_fc(
    const ICiphertext& x_enc,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    int in_features,
    int out_features,
    const ISecretKey& sk,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval
) {
    std::cout << "🔷 FC: " << in_features << " → " << out_features << std::endl;
    
    int n = out_features;
    int log_slots = sk.logDegree() - 1;
    int num_slots = 1 << log_slots;
    
    if (n > num_slots) {
        throw std::runtime_error("out_features > num_slots");
    }
    
    // ------------------------------------------------------------
    // 1. Reformater la matrice poids
    // ------------------------------------------------------------
    std::vector<std::vector<double>> U(out_features, std::vector<double>(in_features, 0.0));
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            U[i][j] = weight[i * in_features + j];
        }
    }
    
    // ------------------------------------------------------------
    // 2. Paramètres BSGS
    // ------------------------------------------------------------
    int n1 = (int)std::sqrt(n);
    int n2 = n / n1;
    while (n1 * n2 < n) n2++;
    while (n1 * n2 > n) n1--;
    
    std::cout << "    BSGS: " << n << " = " << n1 << " × " << n2 << std::endl;
    
    EnDecoder encoder(PresetParamsId::F16Opt_Gr);
    
    // ------------------------------------------------------------
    // 3. Baby steps (i = 1..n2-1)
    // ------------------------------------------------------------
    std::vector<Ptr<ICiphertext>> baby_steps(n2);
    
    for (int i = 1; i < n2; ++i) {
        auto it = rot_keys.find(i);
        if (it == rot_keys.end()) {
            std::cerr << "    ERREUR: Clé rotation " << i << " non trouvée!" << std::endl;
            continue;
        }
        
        auto ct_rot = ICiphertext::make();
        eval.rot(x_enc, i, *ct_rot, *(it->second));
        baby_steps[i] = std::move(ct_rot);
    }
    
    // ------------------------------------------------------------
    // 4. Masque pour extraction (1,0,0,...)
    // ------------------------------------------------------------
    Message<Complex> msg_mask(log_slots, Device::CPU);
    msg_mask[0] = Complex(1.0, 0.0);
    for (int i = 1; i < num_slots; ++i) msg_mask[i] = Complex(0.0, 0.0);
    auto ptxt_mask = IPlaintext::make();
    encoder.encode(msg_mask, *ptxt_mask);
    
    // ------------------------------------------------------------
    // 5. Giant steps
    // ------------------------------------------------------------
    std::vector<Ptr<ICiphertext>> giant_steps(n1);
    
    for (int j = 0; j < n1; ++j) {
        auto ct_gs = ICiphertext::make();
        
        // ---- Cas i = 0 ----
        Message<Complex> msg_diag0(log_slots, Device::CPU);
        for (int k = 0; k < n; ++k) {
            int row = ((-j * n2 + k) % n + n) % n;
            if (row < out_features && k < in_features) {
                msg_diag0[k] = Complex(U[row][k], 0.0);
            } else {
                msg_diag0[k] = Complex(0.0, 0.0);
            }
        }
        
        auto ptxt_diag0 = IPlaintext::make();
        encoder.encode(msg_diag0, *ptxt_diag0);
        
        auto ptxt_diag0_leveled = IPlaintext::make();
        eval.levelDownTo(*ptxt_diag0, *ptxt_diag0_leveled, eval.getLevel(x_enc));
        
        auto ct_mul0 = ICiphertext::make();
        eval.mul(x_enc, *ptxt_diag0_leveled, *ct_mul0);
        eval.rescale(*ct_mul0, *ct_mul0);
        
        // Rotate-and-sum
        auto ct_sum0 = std::move(ct_mul0);
        for (int shift = 1; shift < n; shift <<= 1) {
            auto it_shift = rot_keys.find(shift);
            if (it_shift != rot_keys.end()) {
                auto ct_rot = ICiphertext::make();
                eval.rot(*ct_sum0, shift, *ct_rot, *(it_shift->second));
                
                auto ct_new = ICiphertext::make();
                eval.add(*ct_sum0, *ct_rot, *ct_new);
                ct_sum0 = std::move(ct_new);
            }
        }
        
        // Extraire slot 0
        auto ct_extract0 = ICiphertext::make();
        auto ptxt_mask_leveled0 = IPlaintext::make();
        eval.levelDownTo(*ptxt_mask, *ptxt_mask_leveled0, eval.getLevel(*ct_sum0));
        eval.mul(*ct_sum0, *ptxt_mask_leveled0, *ct_extract0);
        eval.rescale(*ct_extract0, *ct_extract0);
        
        ct_gs = std::move(ct_extract0);
        
        // ---- Cas i = 1..n2-1 ----
        for (int i = 1; i < n2; ++i) {
            if (!baby_steps[i]) continue;
            
            Message<Complex> msg_diag(log_slots, Device::CPU);
            for (int k = 0; k < n; ++k) {
                int row = ((-j * n2 + k) % n + n) % n;
                int col = (k + i) % n;
                if (row < out_features && col < in_features) {
                    msg_diag[k] = Complex(U[row][col], 0.0);
                } else {
                    msg_diag[k] = Complex(0.0, 0.0);
                }
            }
            
            auto ptxt_diag = IPlaintext::make();
            encoder.encode(msg_diag, *ptxt_diag);
            
            auto ptxt_diag_leveled = IPlaintext::make();
            eval.levelDownTo(*ptxt_diag, *ptxt_diag_leveled, eval.getLevel(*baby_steps[i]));
            
            auto ct_mul = ICiphertext::make();
            eval.mul(*baby_steps[i], *ptxt_diag_leveled, *ct_mul);
            eval.rescale(*ct_mul, *ct_mul);
            
            // Rotate-and-sum
            auto ct_sum = std::move(ct_mul);
            for (int shift = 1; shift < n; shift <<= 1) {
                auto it_shift = rot_keys.find(shift);
                if (it_shift != rot_keys.end()) {
                    auto ct_rot = ICiphertext::make();
                    eval.rot(*ct_sum, shift, *ct_rot, *(it_shift->second));
                    
                    auto ct_new = ICiphertext::make();
                    eval.add(*ct_sum, *ct_rot, *ct_new);
                    ct_sum = std::move(ct_new);
                }
            }
            
            // Extraire slot 0
            auto ct_extract = ICiphertext::make();
            auto ptxt_mask_leveled = IPlaintext::make();
            eval.levelDownTo(*ptxt_mask, *ptxt_mask_leveled, eval.getLevel(*ct_sum));
            eval.mul(*ct_sum, *ptxt_mask_leveled, *ct_extract);
            eval.rescale(*ct_extract, *ct_extract);
            
            // Accumulation
            int level_gs = eval.getLevel(*ct_gs);
            int level_extract = eval.getLevel(*ct_extract);
            
            if (level_gs != level_extract) {
                if (level_gs > level_extract) {
                    auto ct_gs_leveled = ICiphertext::make();
                    eval.levelDownTo(*ct_gs, *ct_gs_leveled, level_extract);
                    ct_gs = std::move(ct_gs_leveled);
                } else {
                    auto ct_extract_leveled = ICiphertext::make();
                    eval.levelDownTo(*ct_extract, *ct_extract_leveled, level_gs);
                    ct_extract = std::move(ct_extract_leveled);
                }
            }
            
            auto ct_add = ICiphertext::make();
            eval.add(*ct_gs, *ct_extract, *ct_add);
            ct_gs = std::move(ct_add);
        }
        
        // ---- Rotation géante ----
        if (j > 0) {
            int rot_amount = j * n2;
            auto it = rot_keys.find(rot_amount);
            if (it != rot_keys.end()) {
                auto ct_rot = ICiphertext::make();
                eval.rot(*ct_gs, rot_amount, *ct_rot, *(it->second));
                giant_steps[j] = std::move(ct_rot);
            } else {
                giant_steps[j] = std::move(ct_gs);
            }
        } else {
            giant_steps[j] = std::move(ct_gs);
        }
    }
    
    // ------------------------------------------------------------
    // 6. Sommer tous les giant steps
    // ------------------------------------------------------------
    auto ct_result = ICiphertext::make();
    bool first = true;
    
    for (int j = 0; j < n1; ++j) {
        if (!giant_steps[j]) continue;
        
        if (first) {
            ct_result = std::move(giant_steps[j]);
            first = false;
        } else {
            int level_result = eval.getLevel(*ct_result);
            int level_gs = eval.getLevel(*giant_steps[j]);
            
            if (level_result != level_gs) {
                if (level_result > level_gs) {
                    auto ct_result_leveled = ICiphertext::make();
                    eval.levelDownTo(*ct_result, *ct_result_leveled, level_gs);
                    ct_result = std::move(ct_result_leveled);
                } else {
                    auto ct_gs_leveled = ICiphertext::make();
                    eval.levelDownTo(*giant_steps[j], *ct_gs_leveled, level_result);
                    giant_steps[j] = std::move(ct_gs_leveled);
                }
            }
            
            auto ct_add = ICiphertext::make();
            eval.add(*ct_result, *giant_steps[j], *ct_add);
            ct_result = std::move(ct_add);
        }
    }
    
    // ------------------------------------------------------------
    // 7. Ajouter le bias
    // ------------------------------------------------------------
    Message<Complex> msg_bias(log_slots, Device::CPU);
    for (int i = 0; i < out_features; ++i) {
        msg_bias[i] = Complex(bias[i], 0.0);
    }
    for (int i = out_features; i < num_slots; ++i) {
        msg_bias[i] = Complex(0.0, 0.0);
    }
    
    auto ptxt_bias = IPlaintext::make();
    encoder.encode(msg_bias, *ptxt_bias);
    
    auto ptxt_bias_leveled = IPlaintext::make();
    eval.levelDownTo(*ptxt_bias, *ptxt_bias_leveled, eval.getLevel(*ct_result));
    
    auto ct_add_bias = ICiphertext::make();
    eval.add(*ct_result, *ptxt_bias_leveled, *ct_add_bias);
    ct_result = std::move(ct_add_bias);
    
    std::cout << "    ✅ FC terminé, niveau: " << eval.getLevel(*ct_result) << std::endl;
    
    return ct_result;
}

} // namespace fhe_cnn