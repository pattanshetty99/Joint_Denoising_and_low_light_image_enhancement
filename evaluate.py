import os
import time
import argparse
import cv2
import torch
import lpips
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


# ==========================
# ARGUMENT PARSER
# ==========================

parser = argparse.ArgumentParser(description="Low-Light Enhancement Evaluation")

parser.add_argument("--result_dir", type=str, required=True,
                    help="Directory containing model output images")
parser.add_argument("--gt_dir", type=str, required=True,
                    help="Directory containing ground truth images")
parser.add_argument("--extra_data", type=int, default=0,
                    help="1 if extra training data used, else 0")
parser.add_argument("--save_csv", type=str, default="evaluation_results.csv",
                    help="Output CSV filename")

args = parser.parse_args()

result_dir = args.result_dir
gt_dir = args.gt_dir
extra_data_flag = args.extra_data
csv_name = args.save_csv

# ==========================
# DEVICE SETUP
# ==========================

use_gpu = torch.cuda.is_available()
device_flag = 0 if use_gpu else 1
device = torch.device("cuda" if use_gpu else "cpu")

print("Using device:", device)
print("Device Flag (CPU=1 / GPU=0):", device_flag)
print("Extra Data Flag:", extra_data_flag)

# ==========================
# LPIPS MODEL
# ==========================

lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()

# ==========================
# UTIL FUNCTIONS
# ==========================

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def to_tensor(img):
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    return tensor.to(device)

# ==========================
# EVALUATION LOOP
# ==========================

psnr_list = []
ssim_list = []
lpips_list = []
runtime_list = []

image_names = sorted(os.listdir(result_dir))

print(f"\nEvaluating {len(image_names)} images...\n")

for name in tqdm(image_names):

    result_path = os.path.join(result_dir, name)
    gt_path = os.path.join(gt_dir, name)

    if not os.path.exists(gt_path):
        continue

    # Runtime measurement (image loading only)
    start_time = time.time()

    result_img = load_image(result_path)
    gt_img = load_image(gt_path)

    end_time = time.time()
    runtime_list.append(end_time - start_time)

    # PSNR
    psnr = compare_psnr(gt_img, result_img, data_range=1.0)
    psnr_list.append(psnr)

    # SSIM
    ssim = compare_ssim(gt_img, result_img, channel_axis=2, data_range=1.0)
    ssim_list.append(ssim)

    # LPIPS
    result_tensor = to_tensor(result_img) * 2 - 1
    gt_tensor = to_tensor(gt_img) * 2 - 1

    with torch.no_grad():
        lpips_value = lpips_model(result_tensor, gt_tensor)

    lpips_list.append(lpips_value.item())

# ==========================
# FINAL RESULTS
# ==========================

avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
avg_lpips = np.mean(lpips_list)
avg_runtime = np.mean(runtime_list)

print("\n================ FINAL RESULTS ================")
print(f"PSNR: {avg_psnr:.4f}")
print(f"SSIM: {avg_ssim:.4f}")
print(f"LPIPS: {avg_lpips:.4f}")
print(f"Runtime per image [s]: {avg_runtime:.6f}")
print(f"CPU [1] / GPU [0]: {device_flag}")
print(f"Extra Data [1] / No Extra Data [0]: {extra_data_flag}")
print("================================================")

# ==========================
# SAVE TO CSV
# ==========================

df = pd.DataFrame({
    "PSNR": [avg_psnr],
    "SSIM": [avg_ssim],
    "LPIPS": [avg_lpips],
    "Runtime_per_image_s": [avg_runtime],
    "CPU[1]/GPU[0]": [device_flag],
    "Extra_Data[1]/No_Extra_Data[0]": [extra_data_flag]
})

df.to_csv(csv_name, index=False)

print(f"\nResults saved to {csv_name}")
