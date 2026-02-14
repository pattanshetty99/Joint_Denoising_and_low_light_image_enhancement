import torch
from torch.utils.data import DataLoader
from models.model import LowLightEnhancer_Color
from data.dataset import LowLightDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

lr_train = "path_to_input"
hr_train = "path_to_gt"

dataset = LowLightDataset(lr_train, hr_train)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = LowLightEnhancer_Color().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = torch.nn.L1Loss()

epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)

        optimizer.zero_grad()
        pred = model(lr)
        loss = loss_fn(pred, hr)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "lowlight_model.pth")
