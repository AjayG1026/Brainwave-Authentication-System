import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import spectrogram

# ---------------- CONFIG ----------------
DATA_DIR = "dataset/train"
FS = 256
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
MODEL_PATH = "eeg_auth_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------


class EEGDataset(Dataset):
    def __init__(self, root):
        self.X = []
        self.y = []

        for label, cls in [(1, "genuine"), (0, "impostor")]:
            path = os.path.join(root, cls)
            for f in os.listdir(path):
                x = np.load(os.path.join(path, f))
                self.X.append(x)
                self.y.append(label)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        _, _, Sxx = spectrogram(
            self.X[idx],
            fs=FS,
            nperseg=128,
            noverlap=64
        )
        Sxx = np.log(Sxx + 1e-8)

        X = torch.tensor(Sxx, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return X, y




class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # pool freq only

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # pool freq only

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main():
    dataset = EEGDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EEGNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        epoch_loss = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
