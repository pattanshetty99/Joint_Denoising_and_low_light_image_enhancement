import torch
import cv2
import os
import numpy as np
from models.model import LowLightEnhancer_Color

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LowLightEnhancer_Color().to(device)
checkpoint = torch.load("checkpoints/lowlight_model.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

input_dir = "validation_input"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

for img_name in os.listdir(input_dir):

    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_tensor = torch.tensor(img_rgb).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(img_tensor)

    out = out.squeeze(0).permute(1,2,0).cpu().numpy()
    out = (out * 255).clip(0,255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(save_dir, img_name), out)

print("All images saved.")
