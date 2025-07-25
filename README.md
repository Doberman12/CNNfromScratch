# Implementation of a Convolutional Neural Network Using CuPy

[![Build status](https://github.com/Doberman12/CNNfromScratch/actions/workflows/python-ci.yml/badge.svg)](https://github.com/Doberman12/CNNfromScratch/actions/workflows/python-ci.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f0cf20a8fdb747bbb32a26b6e08c9261)](https://app.codacy.com/gh/Doberman12/CNNfromScratch/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Test coverage](https://codecov.io/github/Doberman12/CNNfromScratch/graph/badge.svg?token=6HYBORPF0L)](https://codecov.io/github/Doberman12/CNNfromScratch)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![Made with CuPy](https://img.shields.io/badge/made%20with-CuPy-1f425f?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
## Table of Contents

- [Overview](#overview)
- [About CuPy](#about-cupy)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [Documentation](#documentation)

---

## Overview

This project presents a complete implementation of a Convolutional Neural Network (CNN) from first principles using the CuPy library, enabling efficient GPU-accelerated computation in Python. It aims to explore the internal workings of CNNs beyond the abstraction provided by widely used frameworks such as TensorFlow or PyTorch.
You can read 

---

## About CuPy

CuPy is an open-source array library for GPU-accelerated computing with Python. It offers a NumPy-compatible interface and leverages NVIDIA CUDA libraries such as cuBLAS, cuDNN, and cuFFT to perform high-speed numerical operations on the GPU. CuPy enables seamless transition from CPU-based NumPy code to GPU execution, making it ideal for scientific computing and deep learning research.

---

## Objectives

- Develop essential CNN components (e.g., convolutional layers, pooling, activation functions, fully connected layers, dropout, and loss functions) as modular Python classes
- Employ optimization strategies like `im2col`, `col2im`, and memory-efficient techniques such as `as_strided` for convolution operations
- Construct a training pipeline using a custom `Sequential` model and implement the AdamOptimizer for gradient-based learning
- Validate the model using the MNIST dataset by assessing classification accuracy, loss, and training time across various batch sizes

---

## Methodology

- Preprocess MNIST images: convert to grayscale `.png` format and normalize input values
- Implement forward and backward propagation without loops, leveraging matrix operations and GPU acceleration
- Analyze model performance through validation curves and confusion matrices

---

## Results

- Achieved maximum test accuracy of **95.36%** with batch size 128
- Performance is comparable to an equivalent TensorFlow implementation in both accuracy and convergence speed
- Training times and memory usage optimized using vectorized operations and strided memory access

---

## Conclusion

This implementation confirms the feasibility of building a performant CNN purely with CuPy, highlighting both computational efficiency and the educational value of low-level deep learning architectures. It provides a foundation for extending the model to more complex datasets and deeper network designs.---

---

## Documentation

A full project report detailing methodology, implementation, results, and analysis is available in the [`DOCS`](./DOCS) folder:

[Project Report (PDF)](./DOCS/CNN_raport.pdf)