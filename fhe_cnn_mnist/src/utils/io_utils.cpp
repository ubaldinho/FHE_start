#include "fhe_cnn/utils.hpp"
#include <fstream>
#include <iostream>
#include <cstdint>

namespace fhe_cnn {

uint32_t read_uint32(std::ifstream &ifs) {
    unsigned char bytes[4];
    ifs.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

std::vector<std::vector<double>> load_mnist_images(const std::string &path) {
    std::cout << "📂 Chargement MNIST images: " << path << std::endl;
    
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open image file: " + path);

    uint32_t magic = read_uint32(ifs);
    uint32_t num_images = read_uint32(ifs);
    uint32_t num_rows = read_uint32(ifs);
    uint32_t num_cols = read_uint32(ifs);

    if (magic != 2051) throw std::runtime_error("Invalid magic number in image file.");

    std::vector<std::vector<double>> images(num_images, 
                                           std::vector<double>(num_rows * num_cols));
    
    for (uint32_t i = 0; i < num_images; ++i) {
        for (uint32_t p = 0; p < num_rows * num_cols; ++p) {
            unsigned char pixel;
            ifs.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][p] = static_cast<double>(pixel) / 255.0;
        }
    }
    
    std::cout << "  ✅ " << num_images << " images chargées, " 
              << num_rows << "×" << num_cols << std::endl;
    
    return images;
}

std::vector<int> load_mnist_labels(const std::string &path) {
    std::cout << "📂 Chargement MNIST labels: " << path << std::endl;
    
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open label file: " + path);

    uint32_t magic = read_uint32(ifs);
    uint32_t num_labels = read_uint32(ifs);

    if (magic != 2049) throw std::runtime_error("Invalid magic number in label file.");

    std::vector<int> labels(num_labels);
    for (uint32_t i = 0; i < num_labels; ++i) {
        unsigned char lbl;
        ifs.read(reinterpret_cast<char*>(&lbl), 1);
        labels[i] = static_cast<int>(lbl);
    }
    
    std::cout << "  ✅ " << num_labels << " labels chargés" << std::endl;
    
    return labels;
}

std::vector<double> load_txt(const std::string& path) {
    std::cout << "📂 Chargement poids: " << path << std::endl;
    
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Failed to open weight file: " + path);
    
    std::vector<double> data;
    double val;
    while (file >> val) {
        data.push_back(val);
    }
    
    std::cout << "  ✅ " << data.size() << " valeurs chargées" << std::endl;
    
    return data;
}

} // namespace fhe_cnn