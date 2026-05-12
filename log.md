# RSP_demo 開發對話紀錄

> 日期：2026-05-12
> 專案：AIoT HW4 — 剪刀石頭布手勢辨識（Raspberry Pi 4）

---

## 對話 1：理解專案結構

**USER：** 理解RSP_demo中所有檔案

**AI：** 分析了 RSP_demo 的完整目錄結構：

```
RSP_demo/
├── README.md              # 刷機教學 + 執行說明 + 評分標準
├── dataset/               # 資料集
│   ├── train/             # 訓練集 (rock / paper / scissors)
│   ├── test/              # 測試集
│   └── validation/        # 驗證集
├── train/                 # 訓練模組
│   ├── train_svm.py       # 訓練 SVM 模型的主程式
│   └── requirements.txt
└── demo/                  # 推論 / Demo 模組
    ├── carema.py          # 攝影機腳本（原始版只是預覽）
    ├── test.py            # 用模型對測試集評估
    ├── rps_svm_model.pkl  # 已訓練的 SVM 模型（~27MB）
    └── requirements.txt
```

**各檔案說明：**

| 檔案 | 功能 |
|------|------|
| `train/train_svm.py` | 讀取 dataset → 轉灰階 → resize(64×64) → flatten → 訓練 SVM(RBF) → 儲存 pkl |
| `demo/test.py` | 載入模型 → 對 dataset/test 進行靜態評估 → 輸出 accuracy / classification report |
| `demo/carema.py` | （原始）只是 OpenCV 攝影機預覽，按 q 離開，無辨識功能 |
| `demo/rps_svm_model.pkl` | 用 joblib 序列化的 SVM 模型（約 27MB） |

---

## 對話 2：確認目前缺少什麼

**USER：** 現在要做什麼事，才能成功執行手勢辨識

**AI：** 盤點現狀後，提出以下行動計畫：

**現狀：**
- `dataset/train/` ✅ 有（rock: 840張+）
- `dataset/test/` ✅ 有（rock: 124張）
- `demo/rps_svm_model.pkl` ✅ 已存在（27MB）
- 即時手勢辨識程式 ❌ **缺少**（carema.py 只是預覽）

**最大缺口：** `carema.py` 需要加入：
1. 載入 `rps_svm_model.pkl`
2. 每幀做前處理（灰階 → 64×64 → flatten → /255.0）
3. 呼叫 `clf.predict()` 得到預測結果
4. 在畫面上顯示 Rock / Paper / Scissors

---

## 對話 3：改寫 carema.py（第一版，原始像素 SVM）

**USER：** 幫我改寫carema.py

**AI：** 改寫 `demo/carema.py`，加入：
- 載入 `rps_svm_model.pkl`
- 前處理管線：`BGR → 灰階 → resize(64×64) → flatten → /255.0`
- 呼叫 `clf.predict()` + `decision_function` 計算信心分數
- UI：頂部半透明條顯示類別 + 信心度；邊框顏色對應類別；`q` 鍵退出
- 自動嘗試攝影機 index 0 和 1

---

## 對話 4：推上 GitHub

**USER：** 幫我把所有檔案推上github `git@github.com:dachengwang-0218/aiot_hw4.git`

**AI：** 執行以下步驟：
1. 確認原有 remote 為 `origin → https://github.com/BiBaIsAFish/RSP_demo.git`
2. 建立 `.gitignore`，排除 `demo/rps_svm_model.pkl`（27MB 超過 GitHub 限制）
3. 新增 remote：`git remote add hw4 git@github.com:dachengwang-0218/aiot_hw4.git`
4. `git add -A && git commit -m "feat: 新增即時手勢辨識功能..."`
5. `git push hw4 main` → **✅ 推送成功**

---

## 對話 5：安裝 cv2（Pylance 報錯）

**USER：** Explain what this problem is and help me fix it: Cannot find module `cv2`

**AI：** 說明這是 **VS Code Pylance 靜態分析錯誤**，非執行期錯誤。

**原因：** Pylance 使用的 Python 3.14 環境中，cv2 缺少型別 stub 檔。

**解法：**
```bash
pip3 install opencv-stubs
```
安裝後 Pylance 即可識別 cv2 型別，紅色底線消失。

---

## 對話 6：手勢辨識無效（關節點不顯示 + 辨識不準）

**USER：** 我執行程式時，沒有手部沒有出現關節點，也沒有根據手勢改變剪刀石頭布

**AI：** 診斷根本原因：

> **問題：** 舊版 SVM 以原始像素（64×64 灰階）訓練，對攝影機的背景 / 光線變化完全沒有泛化能力。且沒有 MediaPipe 關節點偵測。

**解決方案：改用 MediaPipe HandLandmarker**

```
攝影機幀
  └─ MediaPipe HandLandmarker
       ├─ 畫 21 個關節點 + 連線
       └─ 萃取 21×3=63 維座標特徵
              └─ SVM 預測 → Rock / Paper / Scissors
```

**實作步驟：**

1. **下載官方 task 模型：**
   ```bash
   curl -L -o demo/hand_landmarker.task \
     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```

2. **新增 `train/train_mediapipe_svm.py`：**
   - 用 `HandLandmarker` 對 dataset 圖片萃取關節點特徵
   - 以 63 維座標訓練新 SVM（`C=10, kernel='rbf', probability=True`）
   - 儲存至 `demo/rps_mediapipe_svm_model.pkl`

3. **改寫 `demo/carema.py`：**
   - 使用 MediaPipe Tasks API（`RunningMode.IMAGE`）
   - 每幀偵測手部 → 畫關節點連線 → 萃取特徵 → SVM 預測 → 顯示結果

**訓練結果：**

| 類別 | Precision | Recall | F1 |
|------|-----------|--------|----|
| Rock | 0.69 | 1.00 | 0.82 |
| Paper | 1.00 | 0.55 | 0.71 |
| Scissors | 1.00 | 1.00 | 1.00 |
| **Overall** | — | — | **84.95%** |

**注意：** mediapipe 0.10.x 已移除舊版 `mp.solutions` API，需使用新版 `mediapipe.tasks.python.vision.HandLandmarker`。

---

## 對話 7：再次推上 GitHub

**USER：** 幫我推上github

**AI：** 執行：
1. 更新 `.gitignore`，新增排除 `demo/rps_mediapipe_svm_model.pkl`
2. `git add -A`（含 `carema.py`、`hand_landmarker.task`、`train_mediapipe_svm.py`、`.gitignore`）
3. `git commit -m "feat: 改用 MediaPipe HandLandmarker 實現即時手勢辨識"`
4. `git push hw4 main` → **✅ 推送成功**

**本次推送內容：**

| 檔案 | 變更 |
|------|------|
| `demo/carema.py` | 改用 MediaPipe Tasks API，顯示關節點 + SVM 即時辨識 |
| `train/train_mediapipe_svm.py` | 新增：以關節點特徵訓練 SVM |
| `demo/hand_landmarker.task` | 新增：MediaPipe 官方手部偵測模型（7.6MB） |
| `.gitignore` | 排除兩個 `.pkl` 二進位模型檔 |

---

## 執行指令速查

### 訓練（產生新模型）

```bash
cd RSP_demo/train
pip install -r requirements.txt
pip install mediapipe
python3 train_mediapipe_svm.py
```

### Demo（即時辨識）

```bash
cd RSP_demo/demo
python3 carema.py
# 按 q 離開
```

### 靜態測試集評估（舊版像素 SVM）

```bash
cd RSP_demo/demo
python3 test.py
```

---

## 重要技術說明

### MediaPipe API 版本差異

| 版本 | API |
|------|-----|
| < 0.10 | `mp.solutions.hands.Hands()` |
| ≥ 0.10 | `mp.tasks.python.vision.HandLandmarker` + `.task` 模型檔 |

Python 3.14 只支援 mediapipe ≥ 0.10.30，**必須用新版 Tasks API**。

### 特徵設計

```
舊版：BGR → 灰階 → resize(64×64) → flatten(4096 維) → /255.0
新版：BGR → RGB → HandLandmarker → 21 個關節點 × (x,y,z) = 63 維
```

新版特徵對光線、背景、手的大小/位置具有較好的不變性。

### macOS 攝影機授權

```
系統設定 > 隱私權與安全性 > 相機 → 開啟 Terminal 存取
```

---

## 對話 8：建立執行說明文件

**USER：** 給我一個chat.md，裡面說明如何執行carema.py & test.py，一樣放在RSP_demo/

**AI：** 建立 `RSP_demo/chat.md`，內容包含：

| 章節 | 說明 |
|------|------|
| **carema.py** | 前置條件、執行指令、操作說明、macOS 攝影機權限設定 |
| **test.py** | 前置條件、執行指令、範例輸出 |
| **重新訓練** | `train_mediapipe_svm.py` 的執行方式 |
| **兩模型比較表** | 特徵維度、準確率、即時性等差異對比 |

**同期實測結果 — `test.py`（原始像素 SVM）：**

```
總共測試了 372 張圖片
🎯 模型準確率: 68.28%

              precision    recall  f1-score   support
        Rock       0.78      0.63      0.70       124
       Paper       0.61      0.67      0.64       124
    Scissors       0.68      0.75      0.71       124
    accuracy                           0.68       372
```

> 舊版原始像素 SVM 準確率 68.28%，低於 MediaPipe 版（84.95%）。

---

## 對話 9：推上 GitHub 並更新 log.md

**USER：** 幫我推上github，並且更新log.md

**AI：** 更新 `log.md`（新增對話 8、9），一併將以下檔案推上 GitHub：

| 檔案 | 說明 |
|------|------|
| `chat.md` | 新增：執行說明文件 |
| `log.md` | 更新：補充對話 8、9 紀錄 |

```bash
git add chat.md log.md
git commit -m "docs: 新增 chat.md 執行說明，更新 log.md 對話紀錄"
git push hw4 main
```

---

## 對話 10：推薦其他兩個模型

**USER：** 我要使用其他兩個模型，讓test.py準確率提高，請推薦我其他模型

**AI：** 分析問題並推薦以下兩個模型（皆搭配 MediaPipe 63 維關節點特徵）：

| 模型 | 架構說明 | 差異（vs SVM） |
|------|----------|---------------|
| **Random Forest** | 300 棵決策樹投票集成 | 集成學習 vs 最大間距超平面 |
| **MLP 神經網路** | 63 → 256 → 128 → 64 → 3，ReLU + Adam | 多層非線性激活 vs 核函數映射 |

---

## 對話 11：實作兩個新模型

**USER：** 直接幫我實作

**AI：** 建立以下三個新檔案並依序執行訓練：

### 新增檔案

| 檔案 | 說明 |
|------|------|
| `train/train_rf.py` | Random Forest 訓練腳本，`n_estimators=300, class_weight='balanced'` |
| `train/train_mlp.py` | MLP 訓練腳本，架構 63→256→128→64→3，`lr=0.0005, alpha=0.001` |
| `demo/compare_models.py` | 三模型統一評估腳本，輸出排名 |

### 執行方式

```bash
# 訓練
cd RSP_demo/train
python3 train_rf.py
python3 train_mlp.py

# 比較
cd RSP_demo/demo
python3 compare_models.py
```

### 最終比較結果

| 排名 | 模型 | Accuracy | Rock F1 | Paper F1 | Scissors F1 |
|------|------|----------|---------|----------|-------------|
| 🥇 | SVM (MediaPipe) | **84.95%** | 0.82 | 0.71 | 1.00 |
| 🥈 | MLP 神經網路 | 71.51% | 0.77 | 0.27 | 0.88 |
| 🥉 | Random Forest | 52.15% | 0.58 | 0.36 | 0.52 |

### 分析

- **SVM 最佳**：RBF kernel 對低維度結構化特徵（63 維關節點）泛化能力最強
- **RF/MLP 偏低**：訓練集 Rock 有 99 張 MediaPipe 偵測失敗（只剩 741 張），導致類別不平衡，Paper recall 極低

---

## 對話 12：更新 log.md 並推上 GitHub

**USER：** 幫我更新log.md並推上github

**AI：** 更新 `log.md`（新增對話 10、11、12），一併推送以下檔案：

| 檔案 | 說明 |
|------|------|
| `train/train_rf.py` | 新增：Random Forest 訓練腳本 |
| `train/train_mlp.py` | 新增：MLP 訓練腳本 |
| `demo/compare_models.py` | 新增：三模型統一評估腳本 |
| `log.md` | 更新：補充對話 10、11、12 |

```bash
git add train/train_rf.py train/train_mlp.py demo/compare_models.py log.md
git commit -m "feat: 新增 RF / MLP 訓練腳本與三模型比較腳本，更新 log.md"
git push hw4 main
```
