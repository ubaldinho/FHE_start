#include "fhe_cnn/utils.hpp"
#include <iostream>
#include <chrono>

namespace fhe_cnn {

double compute_accuracy(
    const std::vector<int>& predictions,
    const std::vector<int>& labels
) {
    if (predictions.size() != labels.size() || predictions.empty()) {
        return 0.0;
    }
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) correct++;
    }
    
    return 100.0 * correct / predictions.size();
}

void print_timing(const std::chrono::time_point<std::chrono::high_resolution_clock>& start) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  ⏱️  " << duration.count() << " ms" << std::endl;
}

} // namespace fhe_cnn