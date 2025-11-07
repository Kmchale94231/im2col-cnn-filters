# im2col + CNN Filtering (Grayscale Image Convolution)

This project implements 2D convolution using the **im2col** method to convert convolution into fast matrix multiplication (GEMM). I applied several classic image processing filters — Gaussian blur, sharpening, and Sobel edge detection — to a grayscale input image.

---

## Overview

### What `im2col` Does
Normally, convolution slides a kernel across an image. The `im2col` function:
1. Extracts every sliding window (patch) where the kernel "lands"
2. Flattens each patch into a column
3. Stacks all columns into a large matrix

This allows convolution to be computed as:
output = W @ im2col(image)

which is a standard matrix multiplication and much faster in optimized libraries.

### What `conv2d_im2col` Does
- Reshapes filter weights
- Performs matrix multiplication with the `im2col` output
- Reshapes back into an image


## 📁 Project Structure
code/
├── im2col.py # Image → column matrix conversion
├── filters.py # Convolution via im2col + matrix multiply
├── main.py # Runs filters and saves outputs
└── init.py

data/
└── sample.png # Input grayscale image

results/
├── blur.png
├── sharpen.png
├── sobel_x.png
├── sobel_y.png
├── edges.png # Gradient magnitude image
└── grid.png # Combined comparison display

test_im2col.py # Test script for validating im2col correctness

---

## Running the Project

### Install dependencies:
```bash
pip install numpy pillow matplotlib