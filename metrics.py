import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_mse(original, stego):
    original = original.astype(np.float32)
    stego = stego.astype(np.float32)
    return float(np.mean((original - stego) ** 2))

def calculate_psnr(original, stego):
    mse = calculate_mse(original, stego)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    # basic SSIM implementation (grayscale)
    K1, K2 = 0.01, 0.03
    L = 255
    C1, C2 = (K1*L)**2, (K2*L)**2

    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = cv2.GaussianBlur(img1*img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2*img2, (11, 11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return float(ssim_map.mean())

def evaluate_performance(original, stego):
    return {
        "MSE":  calculate_mse(original, stego),
        "PSNR": calculate_psnr(original, stego),
        "SSIM": calculate_ssim(original, stego),
    }

def plot_metrics(results):
    labels, values = list(results.keys()), list(results.values())
    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    plt.title("Performance Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
