"""
train_mlp.py — MLP（多層感知器）訓練腳本
特徵：MediaPipe HandLandmarker 21 關節點 × (x,y,z) = 63 維
架構：63 → 256 → 128 → 64 → 3 (ReLU + Adam)
"""
import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR   = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR    = os.path.join(BASE_DIR, 'dataset', 'test')
TASK_FILE   = os.path.join(BASE_DIR, 'demo', 'hand_landmarker.task')
OUTPUT_PATH = os.path.join(BASE_DIR, 'demo', 'rps_mlp_model.pkl')
LABEL_MAP   = {'rock': 0, 'paper': 1, 'scissors': 2}

if not os.path.exists(TASK_FILE):
    raise FileNotFoundError(f"找不到 hand_landmarker.task：{TASK_FILE}")

# ── HandLandmarker 初始化 ─────────────────────────────────────────────────────
def make_landmarker():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK_FILE),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)

def extract_landmarks(bgr_img, landmarker):
    rgb    = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    return np.array([[p.x, p.y, p.z] for p in lm]).flatten()

# ── 讀取資料集 ────────────────────────────────────────────────────────────────
def load_dataset(folder_path, landmarker):
    X, y = [], []
    for category, label_idx in LABEL_MAP.items():
        cat_path = os.path.join(folder_path, category)
        if not os.path.exists(cat_path):
            subdirs = [d for d in os.listdir(folder_path)
                       if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs:
                cat_path = os.path.join(folder_path, subdirs[0], category)
        if not os.path.exists(cat_path):
            print(f"⚠️  找不到：{cat_path}")
            continue
        files = [f for f in os.listdir(cat_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📂 {category:10s}：{len(files)} 張", end=' ', flush=True)
        s, f = 0, 0
        for fname in files:
            img  = cv2.imread(os.path.join(cat_path, fname))
            if img is None:
                continue
            feat = extract_landmarks(img, landmarker)
            if feat is not None:
                X.append(feat); y.append(label_idx); s += 1
            else:
                f += 1
        print(f"→ 成功 {s} / 失敗 {f}")
    return np.array(X), np.array(y)

# ── 主程式 ─────────────────────────────────────────────────────────────────────
def main():
    landmarker = make_landmarker()

    print("=== 步驟 1：讀取訓練集 ===")
    X_train, y_train = load_dataset(TRAIN_DIR, landmarker)
    print(f"   → 共 {len(X_train)} 筆\n")

    print("=== 步驟 2：讀取測試集 ===")
    X_test, y_test = load_dataset(TEST_DIR, landmarker)
    print(f"   → 共 {len(X_test)} 筆\n")

    landmarker.close()

    if len(X_train) == 0:
        print("❌ 無法萃取任何特徵，請確認 dataset 目錄結構。")
        return

    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)

    print("=== 步驟 3：訓練 MLP（神經網路）===")
    print("   架構：63 → 256 → 128 → 64 → 3，ReLU + Adam optimizer")
    # Pipeline：先 StandardScaler 再 MLP（MLP 對輸入尺度敏感）
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # 三個隱藏層
            activation='relu',
            solver='adam',
            learning_rate_init=0.0005,   # 較小的學習率确保收斂
            alpha=0.001,                 # L2 正則化防止 overfitting
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=30,         # 連續 30 epoch 不提升才停止
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        ))
    ])
    clf.fit(X_train, y_train, mlp__sample_weight=sample_weights)
    mlp = clf.named_steps['mlp']
    print(f"   實際迭代次數：{mlp.n_iter_} 次\n")

    print("=== 步驟 4：評估模型 ===")
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 測試準確率：{accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred,
                                target_names=['Rock', 'Paper', 'Scissors']))

    print("=== 步驟 5：儲存模型 ===")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump(clf, OUTPUT_PATH)
    print(f"✅ 模型儲存至：{OUTPUT_PATH}")

if __name__ == '__main__':
    main()
