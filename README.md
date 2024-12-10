# Image Compression with Wavelet Transform

This project implements a basic image compression and decompression system using wavelet transforms in Python. It supports applying wavelet-based compression to grayscale images, saving compressed data, reconstructing images, and analyzing compression performance through metrics like PSNR (Peak Signal-to-Noise Ratio) and compression ratio.

## Features
- **Wavelet Transform**: Splits image data into low-frequency (approximation) and high-frequency (detail) components.
- **Compression**: High-frequency components below a threshold are zeroed out to achieve compression.
- **Decompression**: Reconstructs the image from compressed wavelet components.
- **Visualization**: Displays original, compressed, and decompressed images.
- **Analysis**: Calculates PSNR and compression ratio for performance evaluation.
- **Graphing**: Visualizes PSNR and compression ratios across multiple images.

## File Structure
- **main.py**: Main script containing all functions for compression, decompression, and analysis.
- **assets/**: Directory containing example images for testing.

## Requirements
This project uses Python 3. Install the following libraries before running:
- `numpy`
- `Pillow` (PIL fork)
- `matplotlib`

You can install the dependencies using pip:
```bash
pip install numpy pillow matplotlib
