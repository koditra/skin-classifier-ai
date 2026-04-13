import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from dataset import SkinDataset, get_transforms
from model import SkinModel
from utils.metrics import compute_metrics

device = "cuda"

DATA_DIR = "train"

classes = sorted(os.listdir(DATA_DIR))
class_to_idx = {c:i for i,c in enumerate(classes)}
idx_to_class = {v:k for k,v in class_to_idx.items()}

dataset = SkinDataset(DATA_DIR, class_to_idx, transform=get_transforms())
loader = torch.utils.data.DataLoader(dataset, batch_size=64)

model = SkinModel(len(classes)).to(device)

ckpt = torch.load("best_model.pth")
model.load_state_dict(ckpt["model"])
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        out = model(x)

        preds = out.argmax(1)

        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())

acc, report, cm = compute_metrics(y_true, y_pred, classes)

print("Accuracy:", acc)
print(report)

disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot(xticks_rotation=90)
plt.show()