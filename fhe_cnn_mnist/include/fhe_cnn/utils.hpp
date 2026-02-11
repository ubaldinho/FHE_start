#ifndef FHE_CNN_UTILS_HPP
#define FHE_CNN_UTILS_HPP

#include <HEAAN2/HEAAN2.hpp>
#include <vector>
#include <string>
#include <map>
#include <chrono>

namespace fhe_cnn {

// ------------------------------------------------------------
// IO Utils
// ------------------------------------------------------------
std::vector<std::vector<double>> load_mnist_images(const std::string& path);
std::vector<int> load_mnist_labels(const std::string& path);
std::vector<double> load_txt(const std::string& path);

// ------------------------------------------------------------
// Packing Utils
// ------------------------------------------------------------
heaan::Message<heaan::Complex> encode_image(
    const std::vector<double>& image,
    int log_slots,
    heaan::Device device
);

heaan::Ptr<heaan::ICiphertext> encrypt_image(
    const std::vector<double>& image,
    const heaan::ISecretKey& sk,
    heaan::EnDecoder& encoder,
    heaan::EnDecryptor& encryptor
);

std::vector<double> decrypt_result(
    const heaan::ICiphertext& ctxt,
    const heaan::ISecretKey& sk,
    heaan::EnDecoder& encoder,
    heaan::EnDecryptor& encryptor,
    int n
);

// ------------------------------------------------------------
// Key Utils
// ------------------------------------------------------------
void generate_all_rot_keys(
    const heaan::ISecretKey& sk,
    int max_rot,
    std::map<int, heaan::Ptr<heaan::ISwKey>>& rot_keys
);

// ------------------------------------------------------------
// Scaling Utils
// ------------------------------------------------------------
double compute_scale_factor(const std::vector<double>& activations);
heaan::Ptr<heaan::ICiphertext> scale_ciphertext(
    const heaan::ICiphertext& ctxt,
    double factor,
    heaan::HomEval& eval
);

// ------------------------------------------------------------
// Metrics Utils
// ------------------------------------------------------------
double compute_accuracy(
    const std::vector<int>& predictions,
    const std::vector<int>& labels
);

void print_timing(const std::chrono::time_point<std::chrono::high_resolution_clock>& start);

} // namespace fhe_cnn

#endif // FHE_CNN_UTILS_HPP