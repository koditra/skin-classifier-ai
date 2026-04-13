
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from sklearn.metrics import f1_score

from model import SkinModel
from dataset import SkinDataset

# ---------------- CONFIG ----------------
DATA_DIR = "/root/.cache/kagglehub/datasets/vinayjayanti/skin-lesion-image-classification/versions/1/my_data"
train_dir = DATA_DIR + "/train"
val_dir = DATA_DIR + "/val"

BATCH_SIZE = 32
EPOCHS = 30
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- DATA ----------------
train_ds = SkinDataset(train_dir, transform=train_transform)
val_ds = SkinDataset(val_dir, transform=val_transform)

# ---------------- CLASS BALANCING ----------------
class_counts = np.bincount(train_ds.labels)
class_weights = 1. / (class_counts + 1e-6)

sample_weights = class_weights[train_ds.labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ---------------- MODEL ----------------
model = SkinModel(num_classes=14).to(device)

# ---------------- LOSS ----------------
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor,
    label_smoothing=0.1
)

# ---------------- OPTIMIZER ----------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# ---------------- AMP (FASTER GPU TRAINING) ----------------
scaler = torch.amp.GradScaler("cuda")
# ---------------- MIXUP ----------------
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

# ---------------- TRAIN LOOP ----------------
best_acc = 0
patience = 5
no_improve = 0

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        imgs, y_a, y_b, lam = mixup(imgs, labels)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(imgs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()

    # ---------------- VALIDATION ----------------
    model.eval()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = np.mean(np.array(preds_all) == np.array(labels_all))
    f1 = f1_score(labels_all, preds_all, average="macro")

    avg_loss = total_loss / len(train_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Val Acc: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # ---------------- SAVE BEST ----------------
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 Saved best model")
        no_improve = 0
    else:
        no_improve += 1

    # ---------------- EARLY STOP ----------------
    if no_improve >= patience:
        print("⛔ Early stopping triggered")
        break