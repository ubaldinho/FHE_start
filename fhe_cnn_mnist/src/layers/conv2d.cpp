#include "fhe_cnn/conv2d.hpp"
#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <cmath>

namespace fhe_cnn {

using namespace heaan;

Ptr<ICiphertext> homomorphic_conv2d(
    const ICiphertext& input_enc,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    int in_c,
    int in_h,
    int in_w,
    int out_c,
    int kernel,
    int out_h,
    int out_w,
    const ISecretKey& sk,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    const ISwKey& relin_key,
    HomEval& eval
) {
    std::cout << "🔷 Conv2D: " << in_c << "×" << in_h << "×" << in_w 
              << " → " << out_c << "×" << out_h << "×" << out_w 
              << ", kernel=" << kernel << std::endl;
    
    int log_slots = sk.logDegree() - 1;
    int num_slots = 1 << log_slots;
    int total_out = out_c * out_h * out_w;
    
    if (total_out > num_slots) {
        throw std::runtime_error("Output size too large for slots");
    }
    
    EnDecoder encoder(PresetParamsId::F16Opt_Gr);
    
    // ------------------------------------------------------------
    // 1. Créer les 25 plaintexts pour les positions du kernel
    //    Chaque plaintext contient les poids pour une position (kh, kw)
    //    répartis sur tous les canaux de sortie
    // ------------------------------------------------------------
    std::vector<Ptr<IPlaintext>> kernel_ptxts;
    
    for (int kh = 0; kh < kernel; ++kh) {
        for (int kw = 0; kw < kernel; ++kw) {
            Message<Complex> msg_kernel(log_slots, Device::CPU);
            
            // Pour chaque canal de sortie et chaque position de sortie
            for (int oc = 0; oc < out_c; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        int slot_idx = (oc * out_h + oh) * out_w + ow;
                        if (slot_idx >= num_slots) continue;
                        
                        // Somme sur les canaux d'entrée
                        double w_sum = 0.0;
                        for (int ic = 0; ic < in_c; ++ic) {
                            int w_idx = (((oc * in_c + ic) * kernel + kh) * kernel + kw);
                            w_sum += weight[w_idx];
                        }
                        msg_kernel[slot_idx] = Complex(w_sum, 0.0);
                    }
                }
            }
            
            auto ptxt = IPlaintext::make();
            encoder.encode(msg_kernel, *ptxt);
            kernel_ptxts.push_back(std::move(ptxt));
        }
    }
    
    // ------------------------------------------------------------
    // 2. Créer les rotations nécessaires de l'image d'entrée
    //    Pour chaque position (kh, kw), on a besoin de Rot_{shift}(input)
    // ------------------------------------------------------------
    std::vector<Ptr<ICiphertext>> rotated_inputs;
    rotated_inputs.push_back(ICiphertext::make());
    *rotated_inputs[0] = input_enc;  // Copie
    
    // Rotations nécessaires pour une image 28×28 avec kernel 5×5
    std::vector<int> shifts = {1, 2, 3, 4, 28, 29, 30, 31, 32, 
                               56, 57, 58, 59, 60, 84, 85, 86, 87, 88};
    
    for (int shift : shifts) {
        auto it = rot_keys.find(shift);
        if (it == rot_keys.end()) {
            std::cerr << "    ERREUR: Clé rotation " << shift << " non trouvée!" << std::endl;
            continue;
        }
        
        auto ct_rot = ICiphertext::make();
        eval.rot(input_enc, shift, *ct_rot, *(it->second));
        rotated_inputs.push_back(std::move(ct_rot));
    }
    
    // ------------------------------------------------------------
    // 3. Pour chaque canal de sortie, calculer la convolution
    // ------------------------------------------------------------
    auto ct_result = ICiphertext::make();
    bool first = true;
    
    for (int oc = 0; oc < out_c; ++oc) {
        std::cout << "    Canal de sortie " << oc << "/" << out_c << std::endl;
        
        auto ct_oc = ICiphertext::make();
        bool first_kernel = true;
        
        // Accumuler les 25 positions du kernel
        for (int kh = 0; kh < kernel; ++kh) {
            for (int kw = 0; kw < kernel; ++kw) {
                // Calculer le shift nécessaire pour cette position
                int shift = kh * in_w + kw;
                
                // Trouver l'image rotatée correspondante
                Ptr<ICiphertext> ct_shifted;
                if (shift == 0) {
                    ct_shifted = ICiphertext::make();
                    *ct_shifted = input_enc;
                } else {
                    auto it = rot_keys.find(shift);
                    if (it != rot_keys.end()) {
                        ct_shifted = ICiphertext::make();
                        eval.rot(input_enc, shift, *ct_shifted, *(it->second));
                    } else {
                        continue;
                    }
                }
                
                // Multiplier par les poids
                int kernel_idx = kh * kernel + kw;
                auto ct_mul = ICiphertext::make();
                eval.mul(*ct_shifted, *kernel_ptxts[kernel_idx], *ct_mul);
                eval.rescale(*ct_mul, *ct_mul);
                
                // Accumuler
                if (first_kernel) {
                    ct_oc = std::move(ct_mul);
                    first_kernel = false;
                } else {
                    int level_oc = eval.getLevel(*ct_oc);
                    int level_mul = eval.getLevel(*ct_mul);
                    
                    if (level_oc != level_mul) {
                        if (level_oc > level_mul) {
                            auto ct_oc_leveled = ICiphertext::make();
                            eval.levelDownTo(*ct_oc, *ct_oc_leveled, level_mul);
                            ct_oc = std::move(ct_oc_leveled);
                        } else {
                            auto ct_mul_leveled = ICiphertext::make();
                            eval.levelDownTo(*ct_mul, *ct_mul_leveled, level_oc);
                            ct_mul = std::move(ct_mul_leveled);
                        }
                    }
                    
                    auto ct_add = ICiphertext::make();
                    eval.add(*ct_oc, *ct_mul, *ct_add);
                    ct_oc = std::move(ct_add);
                }
            }
        }
        
        // Ajouter le bias
        Message<Complex> msg_bias(log_slots, Device::CPU);
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int slot_idx = (oc * out_h + oh) * out_w + ow;
                msg_bias[slot_idx] = Complex(bias[oc], 0.0);
            }
        }
        
        auto ptxt_bias = IPlaintext::make();
        encoder.encode(msg_bias, *ptxt_bias);
        
        auto ptxt_bias_leveled = IPlaintext::make();
        eval.levelDownTo(*ptxt_bias, *ptxt_bias_leveled, eval.getLevel(*ct_oc));
        
        auto ct_add_bias = ICiphertext::make();
        eval.add(*ct_oc, *ptxt_bias_leveled, *ct_add_bias);
        ct_oc = std::move(ct_add_bias);
        
        // Accumuler les canaux de sortie
        if (first) {
            ct_result = std::move(ct_oc);
            first = false;
        } else {
            int level_res = eval.getLevel(*ct_result);
            int level_oc = eval.getLevel(*ct_oc);
            
            if (level_res != level_oc) {
                if (level_res > level_oc) {
                    auto ct_res_leveled = ICiphertext::make();
                    eval.levelDownTo(*ct_result, *ct_res_leveled, level_oc);
                    ct_result = std::move(ct_res_leveled);
                } else {
                    auto ct_oc_leveled = ICiphertext::make();
                    eval.levelDownTo(*ct_oc, *ct_oc_leveled, level_res);
                    ct_oc = std::move(ct_oc_leveled);
                }
            }
            
            auto ct_add = ICiphertext::make();
            eval.add(*ct_result, *ct_oc, *ct_add);
            ct_result = std::move(ct_add);
        }
    }
    
    std::cout << "    ✅ Conv2D terminé, niveau: " << eval.getLevel(*ct_result) << std::endl;
    
    return ct_result;
}

} // namespace fhe_cnn