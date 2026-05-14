"""
train_efficientnet_lite.py — EfficientNet-Lite0 遷移學習訓練腳本（via timm）
架構：EfficientNet-Lite0，移除 Squeeze-Excitation，改用 ReLU6
特點：比 EfficientNet-B0 更適合邊緣裝置推論（無 SE block，更快）
"""
import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR   = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR    = os.path.join(BASE_DIR, 'dataset', 'test')
OUTPUT_PATH = os.path.join(BASE_DIR, 'demo', 'rps_efficientnet_lite_model.pth')

# ── 超參數 ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 3e-4
NUM_CLASSES = 3

# ── 裝置 ──────────────────────────────────────────────────────────────────────
device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('cpu')
)
print(f"🖥️  使用裝置：{device}")

# ── 資料增強 & 正規化 ─────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── 資料集 ────────────────────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_tf)

CLASS_NAMES = train_dataset.classes   # ['paper', 'rock', 'scissors']
print(f"   類別對應：{train_dataset.class_to_idx}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

print(f"   訓練集：{len(train_dataset)} 張，測試集：{len(test_dataset)} 張\n")

# ── 模型建立（遷移學習）───────────────────────────────────────────────────────
print("=== 步驟 1：建立 EfficientNet-Lite0 模型（ImageNet 預訓練）===")
# timm 提供 efficientnet_lite0，pretrained=True 自動下載權重
model = timm.create_model(
    'efficientnet_lite0',
    pretrained=True,
    num_classes=NUM_CLASSES,
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"   總參數量：{total_params:,}  ({total_params/1e6:.2f}M)\n")

# ── 訓練設定 ──────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── 訓練迴圈 ──────────────────────────────────────────────────────────────────
print("=== 步驟 2：開始訓練 ===")
best_acc  = 0.0
best_path = OUTPUT_PATH.replace('.pth', '_best.pth')

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    train_acc  = correct / total * 100
    epoch_loss = total_loss / total
    scheduler.step()

    # 驗證
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total   += imgs.size(0)

    val_acc = val_correct / val_total * 100

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({'model_state': model.state_dict(),
                    'class_to_idx': train_dataset.class_to_idx,
                    'classes': CLASS_NAMES}, best_path)

    print(f"   Epoch [{epoch:2d}/{EPOCHS}]  "
          f"loss: {epoch_loss:.4f}  "
          f"train: {train_acc:.1f}%  "
          f"val: {val_acc:.1f}%"
          + ("  ← best" if val_acc == best_acc else ""))

# ── 最終評估 ──────────────────────────────────────────────────────────────────
print(f"\n=== 步驟 3：載入最佳模型評估（val_acc={best_acc:.2f}%）===")
ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"🎯 最終測試準確率：{accuracy * 100:.2f}%\n")
print(classification_report(all_labels, all_preds,
                            target_names=CLASS_NAMES, digits=4))

# ── 儲存最終模型 ──────────────────────────────────────────────────────────────
print("=== 步驟 4：儲存模型 ===")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save({'model_state': model.state_dict(),
            'class_to_idx': train_dataset.class_to_idx,
            'classes': CLASS_NAMES}, OUTPUT_PATH)
os.remove(best_path)
print(f"✅ 模型儲存至：{OUTPUT_PATH}")
