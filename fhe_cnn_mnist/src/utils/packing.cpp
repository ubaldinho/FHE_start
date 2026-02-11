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

} // namespace fhe_cnn