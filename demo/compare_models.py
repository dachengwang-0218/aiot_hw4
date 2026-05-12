"""
compare_models.py — 三模型統一評估腳本
對 dataset/test 進行批次評估，比較 SVM / Random Forest / MLP 的表現。
"""
import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.metrics import accuracy_score, classification_report

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
DEMO_DIR  = os.path.dirname(__file__)
BASE_DIR  = os.path.dirname(DEMO_DIR)
TEST_DIR  = os.path.join(BASE_DIR, 'dataset', 'test')
TASK_FILE = os.path.join(DEMO_DIR, 'hand_landmarker.task')
LABEL_MAP = {'rock': 0, 'paper': 1, 'scissors': 2}

MODELS = {
    'SVM  (MediaPipe)':           os.path.join(DEMO_DIR, 'rps_mediapipe_svm_model.pkl'),
    'Random Forest (MediaPipe)':  os.path.join(DEMO_DIR, 'rps_rf_model.pkl'),
    'MLP  (MediaPipe)':           os.path.join(DEMO_DIR, 'rps_mlp_model.pkl'),
}

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

# ── 讀取測試集 ────────────────────────────────────────────────────────────────
def load_test_set(landmarker):
    X, y = [], []
    for category, label_idx in LABEL_MAP.items():
        cat_path = os.path.join(TEST_DIR, category)
        if not os.path.exists(cat_path):
            subdirs = [d for d in os.listdir(TEST_DIR)
                       if os.path.isdir(os.path.join(TEST_DIR, d))]
            if subdirs:
                cat_path = os.path.join(TEST_DIR, subdirs[0], category)
        if not os.path.exists(cat_path):
            print(f"⚠️  找不到：{cat_path}")
            continue
        for fname in os.listdir(cat_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img  = cv2.imread(os.path.join(cat_path, fname))
            if img is None:
                continue
            feat = extract_landmarks(img, landmarker)
            if feat is not None:
                X.append(feat)
                y.append(label_idx)
    return np.array(X), np.array(y)

# ── 主程式 ─────────────────────────────────────────────────────────────────────
def main():
    # 確認 task 檔存在
    if not os.path.exists(TASK_FILE):
        print(f"❌ 找不到 hand_landmarker.task：{TASK_FILE}")
        return

    print("⏳ 讀取測試集（使用 MediaPipe 萃取特徵）...")
    landmarker = make_landmarker()
    X_test, y_test = load_test_set(landmarker)
    landmarker.close()
    print(f"✅ 共讀取 {len(X_test)} 筆測試樣本\n")

    if len(X_test) == 0:
        print("❌ 沒有讀到任何測試資料。")
        return

    sep = "=" * 60
    results = {}

    for model_name, model_path in MODELS.items():
        print(sep)
        print(f"  模型：{model_name}")
        print(sep)

        if not os.path.exists(model_path):
            print(f"  ⚠️  找不到模型檔：{model_path}")
            print(f"  請先執行對應的訓練腳本。\n")
            continue

        clf    = joblib.load(model_path)
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        results[model_name] = acc

        print(f"  🎯 準確率：{acc * 100:.2f}%\n")
        print(classification_report(y_test, y_pred,
                                    target_names=['Rock', 'Paper', 'Scissors'],
                                    digits=4))

    # 最終排名
    if results:
        print(sep)
        print("  📊 模型準確率排名")
        print(sep)
        for rank, (name, acc) in enumerate(
            sorted(results.items(), key=lambda x: x[1], reverse=True), 1
        ):
            print(f"  #{rank}  {name:<30s}  {acc*100:.2f}%")
        print()

if __name__ == '__main__':
    main()
