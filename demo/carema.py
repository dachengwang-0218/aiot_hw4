import cv2
import numpy as np
import joblib
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── 設定 ──────────────────────────────────────────────────────────────────────
DEMO_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(DEMO_DIR, 'rps_mediapipe_svm_model.pkl')
TASK_FILE   = os.path.join(DEMO_DIR, 'hand_landmarker.task')
LABEL_NAMES = {0: 'Rock  🪨', 1: 'Paper  📄', 2: 'Scissors  ✂️'}
LABEL_COLOR = {
    0: (60,  60,  220),   # Rock     → 紅
    1: (50,  200,  50),   # Paper    → 綠
    2: (220, 180,   0),   # Scissors → 藍(橘黃)
}

# MediaPipe 連線定義（用來手動畫連線）
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # 拇指
    (0,5),(5,6),(6,7),(7,8),          # 食指
    (0,9),(9,10),(10,11),(11,12),     # 中指
    (0,13),(13,14),(14,15),(15,16),   # 無名指
    (0,17),(17,18),(18,19),(19,20),   # 小指
    (5,9),(9,13),(13,17),             # 掌心橫線
]

# ── 安全性檢查 ────────────────────────────────────────────────────────────────
for path, name in [(MODEL_PATH, 'SVM 模型'), (TASK_FILE, 'hand_landmarker.task')]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到 {name}：{path}")

print("⏳ 載入模型...")
clf = joblib.load(MODEL_PATH)
print("✅ 模型載入完成！按 'q' 離開。\n")

# ── HandLandmarker 初始化（即時影片模式）─────────────────────────────────────
opts = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=TASK_FILE),
    running_mode=mp_vision.RunningMode.IMAGE,   # 每幀獨立偵測，不需 timestamp
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.HandLandmarker.create_from_options(opts)

# ── 特徵萃取 ──────────────────────────────────────────────────────────────────
def extract_landmarks(landmarks_list) -> np.ndarray:
    """MediaPipe landmark list → 63 維特徵向量"""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list]).flatten()

# ── 畫關節點與連線（不依賴 mp.solutions.drawing_utils）──────────────────────
def draw_landmarks(frame, lm_norm, h, w, color):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_norm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        dot_color = color if i == 0 else (255, 255, 255)
        cv2.circle(frame, (px, py), 6, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 6, (50, 50, 50), 1, cv2.LINE_AA)

# ── UI 疊加 ────────────────────────────────────────────────────────────────────
def draw_ui(frame, label_idx, confidence):
    h, w  = frame.shape[:2]
    color = LABEL_COLOR[label_idx]
    name  = LABEL_NAMES[label_idx]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"{name}   {confidence*100:.1f}%",
                (16, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to quit",
                (w - 220, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)

# ── 主迴圈 ────────────────────────────────────────────────────────────────────
def main():
    import platform
    backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else cv2.CAP_V4L2

    cap = None
    for idx in [0, 1]:
        c = cv2.VideoCapture(idx, backend)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                cap = c
                print(f"✅ 使用攝影機 index={idx}")
                break
            c.release()

    if cap is None:
        print("❌ 無法開啟攝影機。")
        print("   macOS：系統設定 > 隱私與安全 > 相機 → 允許 Terminal")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("📷 攝影機已啟動，開始辨識...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)          # 鏡像翻轉
        h, w  = frame.shape[:2]

        # MediaPipe 偵測
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        if result.hand_landmarks:
            lm_norm = result.hand_landmarks[0]   # 21 個正規化座標

            # ① 萃取特徵 → 預測
            feat      = extract_landmarks(lm_norm).reshape(1, -1)
            label_idx = int(clf.predict(feat)[0])

            try:
                proba      = clf.predict_proba(feat)[0]
                confidence = float(proba[label_idx])
            except Exception:
                confidence = 1.0

            color = LABEL_COLOR[label_idx]

            # ② 畫關節點與連線
            draw_landmarks(frame, lm_norm, h, w, color)

            # ③ 疊加 UI
            draw_ui(frame, label_idx, confidence)

        else:
            cv2.putText(frame, "No hand detected — show your hand!",
                        (16, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (100, 100, 100), 2, cv2.LINE_AA)

        cv2.imshow("RPS Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("👋 程式結束。")

if __name__ == "__main__":
    main()