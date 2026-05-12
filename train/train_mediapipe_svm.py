"""
train_mediapipe_svm.py  (mediapipe >= 0.10.x Tasks API)
用 MediaPipe HandLandmarker 偵測手部 21 個關節點，
以 21×3=63 個座標值作為 SVM 特徵重新訓練。
"""
import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ── 設定 ──────────────────────────────────────────────────────────────────────
LABEL_MAP    = {'rock': 0, 'paper': 1, 'scissors': 2}
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR    = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR     = os.path.join(BASE_DIR, 'dataset', 'test')
MODEL_FILE   = os.path.join(BASE_DIR, 'demo', 'hand_landmarker.task')
OUTPUT_PATH  = os.path.join(BASE_DIR, 'demo', 'rps_mediapipe_svm_model.pkl')

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"找不到 {MODEL_FILE}\n"
        "請先執行：\n"
        "  curl -L -o demo/hand_landmarker.task \\\n"
        "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

# ── HandLandmarker 初始化（靜態圖片模式）─────────────────────────────────────
def make_landmarker():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)

def extract_landmarks(bgr_img, landmarker) -> np.ndarray | None:
    """BGR 圖片 → 63 維特徵向量（21 landmarks × x,y,z）；偵測失敗回傳 None"""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]          # 第一隻手的 21 個點
    return np.array([[p.x, p.y, p.z] for p in lm]).flatten()

# ── 讀取資料集 ────────────────────────────────────────────────────────────────
def load_dataset(folder_path, landmarker):
    X, y = [], []
    success, failed = 0, 0

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

        for fname in files:
            img = cv2.imread(os.path.join(cat_path, fname))
            if img is None:
                continue
            feat = extract_landmarks(img, landmarker)
            if feat is not None:
                X.append(feat)
                y.append(label_idx)
                success += 1
            else:
                failed += 1

        print(f"→ 成功 {success} / 失敗 {failed}")
        success, failed = 0, 0

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

    print("=== 步驟 3：訓練 SVM（RBF kernel）===")
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    clf.fit(X_train, y_train)
    print("   訓練完成！\n")

    if len(X_test) > 0:
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
