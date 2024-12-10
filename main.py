import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image


psnrs_values = []
compression_values = []


def display_image(img_array, title="Image"):
    plt.imshow(img_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def apply_wavelet_transform(img_array):
    rows, cols = img_array.shape
    if rows % 2 != 0:
        img_array = img_array[:rows-1, :]
    if cols % 2 != 0:
        img_array = img_array[:, :cols-1]

    rows, cols = img_array.shape
    low_freq = ((img_array[0:rows:2, 0:cols:2] +
                img_array[1:rows:2, 0:cols:2] +
                img_array[0:rows:2, 1:cols:2] +
                img_array[1:rows:2, 1:cols:2])/4).astype(np.float64)
    high_freq_h = ((img_array[0:rows:2, 0:cols:2] - img_array[1:rows:2, 0:cols:2])/2).astype(np.float64)
    high_freq_v = ((img_array[0:rows:2, 0:cols:2] - img_array[0:rows:2, 1:cols:2])/2).astype(np.float64)
    high_freq_d = ((img_array[0:rows:2, 0:cols:2] - img_array[1:rows:2, 1:cols:2])/2).astype(np.float64)

    return low_freq, high_freq_h, high_freq_v, high_freq_d


def compress_image(input_path, output_path, threshold=10):
    img = Image.open(input_path).convert('L')
    img_array = np.array(img, dtype=np.float64)
    display_image(img_array, title="Original Image")

    low_freq, high_freq_h, high_freq_v, high_freq_d = apply_wavelet_transform(img_array)
    high_freq_h[np.abs(high_freq_h) < threshold] = 0
    high_freq_v[np.abs(high_freq_v) < threshold] = 0
    high_freq_d[np.abs(high_freq_d) < threshold] = 0

    np.savez_compressed(output_path, low_freq=low_freq, high_freq_h=high_freq_h,
                        high_freq_v=high_freq_v, high_freq_d=high_freq_d)
    print(f"Compressed image saved as {output_path}.npz")

    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(f"{output_path}.npz")
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio: {compression_ratio:.2f}")

    compression_values.append(compression_ratio)

    return img_array


def reconstruct_wavelet_transform(low_freq, high_freq_h, high_freq_v, high_freq_d):
    rows, cols = low_freq.shape
    reconstructed_img = np.zeros((rows * 2, cols * 2), dtype=np.float64)

    reconstructed_img[0:rows*2:2, 0:cols*2:2] = low_freq + high_freq_h + high_freq_v + high_freq_d
    reconstructed_img[1:rows*2:2, 0:cols*2:2] = low_freq - high_freq_h + high_freq_v - high_freq_d
    reconstructed_img[0:rows*2:2, 1:cols*2:2] = low_freq + high_freq_h - high_freq_v - high_freq_d
    reconstructed_img[1:rows*2:2, 1:cols*2:2] = low_freq - high_freq_h - high_freq_v + high_freq_d

    return reconstructed_img


def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def decompress_image(input_path, output_path, original_path):
    compressed_data = np.load(input_path)
    low_freq = compressed_data['low_freq']
    high_freq_h = compressed_data['high_freq_h']
    high_freq_v = compressed_data['high_freq_v']
    high_freq_d = compressed_data['high_freq_d']

    img_array = reconstruct_wavelet_transform(low_freq, high_freq_h, high_freq_v, high_freq_d)
    img_array = np.clip(img_array, 0, 255)

    display_image(img_array, title="Decompressed Image")

    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(output_path)
    print(f"Decompressed image saved as {output_path}")

    original_img = np.array(Image.open(original_path).convert('L'), dtype=np.float64)

    if original_img.shape != img_array.shape:
        min_rows = min(original_img.shape[0], img_array.shape[0])
        min_cols = min(original_img.shape[1], img_array.shape[1])
        original_img = original_img[:min_rows, :min_cols]
        img_array = img_array[:min_rows, :min_cols]

    psnr_value = calculate_psnr(original_img, img_array)
    print(f"PSNR between original and decompressed images: {psnr_value:.2f} dB")

    psnrs_values.append(psnr_value)


def plots():
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(psnrs_values, marker='o', color='b', linestyle='-', label='PSNR')
    ax[0].set_title("PSNR for various images")
    ax[0].set_xlabel("Images")
    ax[0].set_ylabel("PSNR (dB)")
    ax[0].legend()

    ax[1].plot(compression_values, marker='s', color='r', linestyle='-', label='Compression Ratio')
    ax[1].set_title("Compression ratio for various images")
    ax[1].set_xlabel("Images")
    ax[1].set_ylabel("Compression Ratio")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    compress_image('assets\\image.png', r'assets\\compressed_image', threshold=1)
    decompress_image(r'assets\\compressed_image.npz', 'assets\\decompressed_image.png', 'assets\\image.png')

    plots()


if __name__ == "__main__":
    main()
