# Joint_Denoising_and_low_light_image_enhancement

This project implements a transformer-based low-light image enhancement network with:

- Retinex-based illumination correction
- Window Self-Attention Transformer blocks
- Denoising branch
- Color correction branch
- Advanced multi-loss training (L1 + LPIPS + SSIM + Edge + Color)

The model is designed for research-level experiments and CVPR/NTIRE-style evaluation.

---

# Project Structure

```
lowlight_project/
│
├── models/
│   ├── transformer.py
│   ├── blocks.py
│   └── model.py
│
├── data/
│   └── dataset.py
│
├── utils/
│   └── losses.py
│
├── train.py
├── test.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the repository

```
git clone <your_repo_link>
cd Joint_Denoising_and_low_light_image_enhancement
```

## 2. Create virtual environment (recommended)

### Windows
```
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac
```
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

If using CPU:

```
pip install -r requirements.txt
```

If using GPU, install PyTorch first from https://pytorch.org, then:

```
pip install -r requirements.txt --no-deps
```

---

# Dataset Structure

Your dataset must follow this structure:

```
dataset/
│
├── train/
│   ├── input/
│   └── gt/
│
├── validation/
│   ├── val_in/
│   └── val_gt/
```

Input and ground-truth filenames must match.

---

# Training

Edit `train.py` and set:

```
lr_train = "path_to_train_input"
hr_train = "path_to_train_gt"
```

Run:

```
python train.py
```

The model checkpoint will be saved as:

```
lowlight_model.pth
```

Training includes:

- L1 Loss
- LPIPS Loss
- SSIM Loss
- Edge Loss
- Color Consistency Loss
- Mixed Precision (AMP)
- Cosine LR Scheduler
- Gradient Clipping
- Resume Training Support

---

# Testing (Inference)

Edit `test.py`:

```
input_dir = "path_to_validation_input"
save_dir = "results"
```

Run:

```
python test.py
```

Enhanced images will be saved in:

```
results/
```

---

# Evaluation

To compute PSNR, SSIM, LPIPS, and runtime:

```
python evaluate.py --result_dir results --gt_dir path_to_val_gt
```

Optional flag:

```
--extra_data 1
```

Evaluation output:

- Average PSNR
- Average SSIM
- Average LPIPS
- Runtime per image
- CSV file (evaluation_results.csv)

---

# Model Architecture Overview

The model consists of:

1. Illumination Branch  
   Estimates illumination map for Retinex enhancement.

2. Encoder  
   CNN + Window Self-Attention Transformer blocks.

3. Denoiser  
   Residual noise removal block.

4. Structure Decoder  
   Upsampling reconstruction module.

5. Color Branch  
   Global color correction residual learning.

Final output:

```
Output = Structure + Color + Input
```

---

# Metrics Used

- PSNR
- SSIM
- LPIPS
- Edge Consistency
- Color Consistency

---

# Hardware Requirements

Minimum:
- 8GB RAM
- CPU support

Recommended:
- NVIDIA GPU with 6GB+ VRAM
- CUDA-enabled PyTorch

---

# Resume Training

Set inside `train.py`:

```
resume = True
```

Make sure checkpoint file exists.

---

# Notes

- Images must be normalized to [0,1].
- Model output is clamped to [0,1].
- LPIPS expects input in [-1,1].
- Use batch size according to GPU memory.

---

# Future Improvements

- Multi-GPU training
- EMA model averaging
- Validation during training
- Automatic best-model saving
- NTIRE submission pipeline
- TensorBoard logging
- Faster inference benchmarking

---

# License

This project is for research and educational purposes.
