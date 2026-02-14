import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import LowLightEnhancer_Color
from data.dataset import LowLightDataset
from utils.losses import AdvancedLoss

# ==========================
# CONFIG
# ==========================

lr_train = "path_to_input"
hr_train = "path_to_gt"

epochs = 100
batch_size = 4
learning_rate = 5e-5
save_path = "lowlight_model.pth"

resume = False  # set True to resume

# ==========================
# DEVICE
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================
# DATA
# ==========================

dataset = LowLightDataset(lr_train, hr_train)
loader = DataLoader(dataset, batch_size=batch_size,
                    shuffle=True, num_workers=2, pin_memory=True)

print("Training samples:", len(dataset))

# ==========================
# MODEL
# ==========================

model = LowLightEnhancer_Color().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

criterion = AdvancedLoss(device)

scaler = torch.cuda.amp.GradScaler()

start_epoch = 0

if resume and os.path.exists(save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    print("Resumed from checkpoint")

# ==========================
# TRAIN LOOP
# ==========================

for epoch in range(start_epoch, epochs):

    model.train()
    total_loss = 0

    pbar = tqdm(loader)

    for lr_img, hr_img in pbar:

        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        # Random noise augmentation
        noise_std = torch.rand(1).item() * 0.05
        lr_img = torch.clamp(lr_img + torch.randn_like(lr_img) * noise_std, 0, 1)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            pred = model(lr_img)
            loss, loss_dict = criterion(pred, hr_img)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        pbar.set_description(
            f"Epoch {epoch+1} | "
            f"L1 {loss_dict['L1']:.3f} | "
            f"LPIPS {loss_dict['LPIPS']:.3f} | "
            f"SSIM {loss_dict['SSIM']:.3f}"
        )

    scheduler.step()

    avg_loss = total_loss / len(loader)

    print(f"\nEpoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    print("LR:", scheduler.get_last_lr()[0])

    # Save checkpoint
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, save_path)

print("\nTraining Complete.")
