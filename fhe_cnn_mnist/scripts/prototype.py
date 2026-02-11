#!/usr/bin/env python3
"""
Prototype Python pour validation du CNN MNIST
"""

import numpy as np
import struct
from pathlib import Path

def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols).astype(np.float32) / 255.0
    return images

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num = struct.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_txt(path):
    return np.loadtxt(path)

def conv2d(x, w, b):
    out_c, in_c, kh, kw = w.shape
    in_h, in_w = x.shape
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    y = np.zeros((out_c, out_h, out_w))
    for oc in range(out_c):
        for oh in range(out_h):
            for ow in range(out_w):
                s = b[oc]
                for ic in range(in_c):
                    for kh_ in range(kh):
                        for kw_ in range(kw):
                            s += x[ic, oh+kh_, ow+kw_] * w[oc, ic, kh_, kw_]
                y[oc, oh, ow] = s
    return y

def avgpool2d(x):
    c, h, w = x.shape
    out_h = h // 2
    out_w = w // 2
    y = np.zeros((c, out_h, out_w))
    for ch in range(c):
        for oh in range(out_h):
            for ow in range(out_w):
                y[ch, oh, ow] = np.mean(x[ch, oh*2:oh*2+2, ow*2:ow*2+2])
    return y

def relu(x):
    return np.maximum(x, 0)

def linear(x, w, b):
    return w @ x + b

def main():
    print("🔬 Prototype CNN MNIST")
    print("=====================")
    
    # Chargement données
    print("\n1. Chargement données...")
    images = load_mnist_images("../data/mnist/t10k-images-idx3-ubyte")
    labels = load_mnist_labels("../data/mnist/t10k-labels-idx1-ubyte")
    
    # Chargement poids
    print("2. Chargement poids...")
    conv1_w = load_txt("../data/weights/conv1.weight.txt").reshape(8, 1, 5, 5)
    conv1_b = load_txt("../data/weights/conv1.bias.txt")
    conv2_w = load_txt("../data/weights/conv2.weight.txt").reshape(16, 8, 5, 5)
    conv2_b = load_txt("../data/weights/conv2.bias.txt")
    fc1_w = load_txt("../data/weights/fc1.weight.txt").reshape(128, 256)
    fc1_b = load_txt("../data/weights/fc1.bias.txt")
    fc2_w = load_txt("../data/weights/fc2.weight.txt").reshape(64, 128)
    fc2_b = load_txt("../data/weights/fc2.bias.txt")
    fc3_w = load_txt("../data/weights/fc3.weight.txt").reshape(10, 64)
    fc3_b = load_txt("../data/weights/fc3.bias.txt")
    
    # Test sur 10 images
    num_test = 10
    correct = 0
    
    for idx in range(num_test):
        print(f"\n--- Image {idx} ---")
        
        # Reshape: (28,28) -> (1,28,28)
        x = images[idx].reshape(1, 28, 28)
        
        # Forward
        x = conv2d(x, conv1_w, conv1_b)
        x = relu(x)
        x = avgpool2d(x)
        
        x = conv2d(x, conv2_w, conv2_b)
        x = relu(x)
        x = avgpool2d(x)
        
        x = x.flatten()
        x = linear(x, fc1_w, fc1_b)
        x = relu(x)
        x = linear(x, fc2_w, fc2_b)
        x = relu(x)
        x = linear(x, fc3_w, fc3_b)
        
        pred = np.argmax(x)
        true_label = labels[idx]
        
        if pred == true_label:
            correct += 1
            
        print(f"   Prédiction: {pred}, Vérité: {true_label} -> {'✓' if pred == true_label else '✗'}")
    
    print(f"\n=== Résultats ===")
    print(f"Accuracy: {100 * correct / num_test:.1f}% ({correct}/{num_test})")

if __name__ == "__main__":
    main()