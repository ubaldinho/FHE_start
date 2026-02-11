#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "io.hpp"

// conv2d: (no padding, stride=1)
std::vector<double> conv2d(const std::vector<double>& input, int in_c, int in_h, int in_w,
                          const std::vector<double>& weight, const std::vector<double>& bias,
                          int out_c, int kernel, int out_h, int out_w) {
    std::vector<double> output(out_c * out_h * out_w, 0.0f);
    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                double sum = bias[oc];
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < kernel; ++kh) {
                        for (int kw = 0; kw < kernel; ++kw) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            sum += input[(ic * in_h + ih) * in_w + iw] *
                                   weight[(((oc * in_c) + ic) * kernel + kh) * kernel + kw];
                        }
                    }
                }
                output[(oc * out_h + oh) * out_w + ow] = sum;
            }
        }
    }
    return output;
}

// AvgPool 2x2
std::vector<double> avgpool2d(const std::vector<double>& input, int c, int h, int w) {
    int out_h = h / 2;
    int out_w = w / 2;
    std::vector<double> output(c * out_h * out_w, 0.0f);
    for (int ch = 0; ch < c; ++ch) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                double sum = 0;
                for (int kh = 0; kh < 2; ++kh) {
                    for (int kw = 0; kw < 2; ++kw) {
                        int ih = oh * 2 + kh;
                        int iw = ow * 2 + kw;
                        sum += input[(ch * h + ih) * w + iw];
                    }
                }
                output[(ch * out_h + oh) * out_w + ow] = sum / 4.0f;
            }
        }
    }
    return output;
}

// ReLU
void relu(std::vector<double>& x) {
    for (auto &v : x) if (v < 0) v = 0;
}

// Linear layer
std::vector<double> linear(const std::vector<double>& x,
                          const std::vector<double>& weight,
                          const std::vector<double>& bias,
                          int out_features, int in_features) {
    std::vector<double> y(out_features, 0.0f);
    for (int o = 0; o < out_features; ++o) {
        double sum = bias[o];
        for (int i = 0; i < in_features; ++i) {
            sum += weight[o * in_features + i] * x[i];
        }
        y[o] = sum;
    }
    return y;
}

int main() {
    // 1. Load weights
    auto conv1_w = load_txt("weights/conv1.weight.txt");
    auto conv1_b = load_txt("weights/conv1.bias.txt");
    auto conv2_w = load_txt("weights/conv2.weight.txt");
    auto conv2_b = load_txt("weights/conv2.bias.txt");
    auto fc1_w = load_txt("weights/fc1.weight.txt");
    auto fc1_b = load_txt("weights/fc1.bias.txt");
    auto fc2_w = load_txt("weights/fc2.weight.txt");
    auto fc2_b = load_txt("weights/fc2.bias.txt");
    auto fc3_w = load_txt("weights/fc3.weight.txt");
    auto fc3_b = load_txt("weights/fc3.bias.txt");

    auto images = load_mnist_images("mnist/t10k-images-idx3-ubyte"); // std::vector<std::vector<double>>
    auto labels = load_mnist_labels("mnist/t10k-labels-idx1-ubyte");  // std::vector<int>

    std::cout << "Loading complete !" << std::endl;
    int num_test = images.size();
    int correct = 0;
    int h = 28, w = 28;

    num_test = 100;
    for (int idx = 0; idx < num_test; idx ++) {
        std::vector<double> input = images[idx]; // size 784, 1x28x28

        // 2. Forward pass
        auto x = conv2d(input, 1, h, w, conv1_w, conv1_b, 8, 5, 24, 24);
        relu(x);
        x = avgpool2d(x, 8, 24, 24); // 8x12x12

        x = conv2d(x, 8, 12, 12, conv2_w, conv2_b, 16, 5, 8, 8);
        relu(x);
        x = avgpool2d(x, 16, 8, 8); // 16x4x4

        // flatten
        std::vector<double> flat(x.begin(), x.end()); // 256

        x = linear(flat, fc1_w, fc1_b, 128, 256);
        relu(x);
        x = linear(x, fc2_w, fc2_b, 64, 128);
        relu(x);
        x = linear(x, fc3_w, fc3_b, 10, 64);

        // argmax
        int pred = std::max_element(x.begin(), x.end()) - x.begin();


        if (pred == labels[idx]) correct++;
        std::cout << idx << "/" << num_test << " complete: " << (pred == labels[idx]) << std::endl;
    }

    double acc = 100.0f * correct / num_test;
    std::cout << "Accuracy: " << acc << "% (" << correct << "/" << num_test << ")\n";

    return 0;
}