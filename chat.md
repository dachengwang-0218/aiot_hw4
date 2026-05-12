# 執行說明

## 前置準備

確認已安裝所需套件：

```bash
pip install opencv-python numpy joblib scikit-learn mediapipe
```

---

## carema.py — 即時手勢辨識

### 說明

開啟攝影機，使用 **MediaPipe HandLandmarker** 偵測手部 21 個關節點，
並以 **SVM 模型**即時預測剪刀石頭布（Rock / Paper / Scissors）。

### 前置條件

執行前，`demo/` 資料夾內需有以下兩個檔案：

| 檔案 | 取得方式 |
|------|----------|
| `rps_mediapipe_svm_model.pkl` | 執行 `train/train_mediapipe_svm.py` 產生 |
| `hand_landmarker.task` | 已含於 repo，或用以下指令下載 |

```bash
# 下載 hand_landmarker.task（若不存在）
curl -L -o demo/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 執行

```bash
cd RSP_demo/demo
python3 carema.py
```

### 操作說明

- 將手放在攝影機前，畫面會顯示 **21 個關節點與連線**
- 頂部顯示辨識結果與信心分數（例如：`Rock 🪨  91.3%`）
- 邊框顏色對應類別（紅 = Rock、綠 = Paper、橘黃 = Scissors）
- 按 **`q`** 退出

### macOS 注意事項

若出現攝影機無法開啟：

```
系統設定 > 隱私權與安全性 > 相機 → 打開 Terminal 的存取權限
```

---

## test.py — 靜態測試集評估

### 說明

以 **原始像素 SVM 模型**（`rps_svm_model.pkl`）對 `dataset/test/` 做批次評估，
輸出整體準確率與各類別的 precision / recall / F1-score。

> ⚠️ 此腳本使用**舊版原始像素模型**，準確率約 68%。
> 即時辨識請使用 `carema.py`（MediaPipe 版，準確率 84.95%）。

### 前置條件

- `demo/rps_svm_model.pkl`（原始像素 SVM 模型）
- `dataset/test/rock/`、`dataset/test/paper/`、`dataset/test/scissors/` 三個資料夾

### 執行

```bash
cd RSP_demo/demo
python3 test.py
```

### 範例輸出

```
⏳ 載入模型中...
✅ 模型載入成功！

📂 正在讀取測試集圖片並進行預測...

📊 測試結果統整:
總共測試了 372 張圖片
🎯 模型準確率: 68.28%

📝 分類詳細報告:
              precision    recall  f1-score   support
        Rock       0.78      0.63      0.70       124
       Paper       0.61      0.67      0.64       124
    Scissors       0.68      0.75      0.71       124
    accuracy                           0.68       372
```

---

## 重新訓練模型（MediaPipe 版）

若需在新環境重新訓練：

```bash
cd RSP_demo/train
python3 train_mediapipe_svm.py
```

訓練完成後，模型自動儲存至 `demo/rps_mediapipe_svm_model.pkl`，
即可直接執行 `carema.py`。

---

## 兩個模型比較

| 項目 | `test.py`（原始像素） | `carema.py`（MediaPipe）|
|------|----------------------|------------------------|
| 特徵維度 | 4096（64×64 灰階）| 63（21 關節點 × x,y,z）|
| 測試準確率 | 68.28% | **84.95%** |
| 對光線/背景 | 敏感 | 較不敏感 |
| 關節點顯示 | ❌ | ✅ |
| 即時辨識 | ❌（批次） | ✅（攝影機） |
