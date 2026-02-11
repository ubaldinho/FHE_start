#include "fhe_cnn/utils.hpp"
#include <iostream>

namespace fhe_cnn {

using namespace heaan;

Message<Complex> encode_image(
    const std::vector<double>& image,
    int log_slots,
    Device device
) {
    Message<Complex> msg(log_slots, device);
    int num_slots = 1 << log_slots;
    
    for (int i = 0; i < num_slots && i < (int)image.size(); ++i) {
        msg[i] = Complex(image[i], 0.0);
    }
    for (int i = image.size(); i < num_slots; ++i) {
        msg[i] = Complex(0.0, 0.0);
    }
    
    return msg;
}

Ptr<ICiphertext> encrypt_image(
    const std::vector<double>& image,
    const ISecretKey& sk,
    EnDecoder& encoder,
    EnDecryptor& encryptor
) {
    int log_slots = sk.logDegree() - 1;
    auto msg = encode_image(image, log_slots, sk.device());
    
    auto ptxt = IPlaintext::make();
    encoder.encode(msg, *ptxt);
    
    auto ctxt = ICiphertext::make();
    encryptor.encrypt(*ptxt, sk, *ctxt);
    
    return ctxt;
}

std::vector<double> decrypt_result(
    const ICiphertext& ctxt,
    const ISecretKey& sk,
    EnDecoder& encoder,
    EnDecryptor& encryptor,
    int n
) {
    auto ptxt = IPlaintext::make();
    encryptor.decrypt(ctxt, sk, *ptxt);
    
    Message<Complex> msg;
    encoder.decode(*ptxt, msg);
    msg.to(Device::CPU);
    
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = msg[i].real() / n;  // Division par n !
    }
    
    return result;
}

// À AJOUTER dans packing.cpp

Message<Complex> pack_4_images(
    const std::vector<std::vector<double>>& images,
    int log_slots,
    Device device
) {
    if (images.size() != 4) {
        throw std::runtime_error("pack_4_images: besoin de 4 images");
    }
    
    int num_slots = 1 << log_slots;
    int img_size = 784;  // 28×28
    
    Message<Complex> msg(log_slots, device);
    
    // Image 1: slots 0-783
    for (int i = 0; i < img_size && i < num_slots; ++i) {
        msg[i] = Complex(images[0][i], 0.0);
    }
    
    // Image 2: slots 784-1567
    for (int i = 0; i < img_size && (i + img_size) < num_slots; ++i) {
        msg[i + img_size] = Complex(images[1][i], 0.0);
    }
    
    // Image 3: slots 1568-2351
    for (int i = 0; i < img_size && (i + 2*img_size) < num_slots; ++i) {
        msg[i + 2*img_size] = Complex(images[2][i], 0.0);
    }
    
    // Image 4: slots 2352-3135
    for (int i = 0; i < img_size && (i + 3*img_size) < num_slots; ++i) {
        msg[i + 3*img_size] = Complex(images[3][i], 0.0);
    }
    
    // Remplir le reste avec 0
    for (int i = 4 * img_size; i < num_slots; ++i) {
        msg[i] = Complex(0.0, 0.0);
    }
    
    std::cout << "    📦 4 images packées: " 
              << 4 * img_size << " slots utilisés" << std::endl;
    
    return msg;
}

std::vector<std::vector<double>> unpack_4_results(
    const ICiphertext& ctxt,
    const ISecretKey& sk,
    EnDecoder& encoder,
    EnDecryptor& decryptor,
    int output_size
) {
    // Déchiffrement
    auto ptxt = IPlaintext::make();
    decryptor.decrypt(ctxt, sk, *ptxt);
    
    Message<Complex> msg;
    encoder.decode(*ptxt, msg);
    msg.to(Device::CPU);
    
    int num_slots = 1 << msg.logSlots();
    int stride = output_size;  // Espacement entre les résultats
    
    std::vector<std::vector<double>> results(4);
    
    // Image 1: slots 0..output_size-1
    results[0].resize(output_size);
    for (int i = 0; i < output_size; ++i) {
        results[0][i] = msg[i].real();
    }
    
    // Image 2: slots stride..stride+output_size-1
    results[1].resize(output_size);
    for (int i = 0; i < output_size; ++i) {
        results[1][i] = msg[i + stride].real();
    }
    
    // Image 3: slots 2*stride..2*stride+output_size-1
    results[2].resize(output_size);
    for (int i = 0; i < output_size; ++i) {
        results[2][i] = msg[i + 2*stride].real();
    }
    
    // Image 4: slots 3*stride..3*stride+output_size-1
    results[3].resize(output_size);
    for (int i = 0; i < output_size; ++i) {
        results[3][i] = msg[i + 3*stride].real();
    }
    
    return results;
}

} // namespace fhe_cnn