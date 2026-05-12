import cv2
import numpy as np
import joblib
import os

# ── 設定 ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'rps_svm_model.pkl')
IMG_SIZE    = 64          # 必須與訓練時一致
LABEL_NAMES = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
LABEL_EMOJI = {0: '🪨', 1: '📄', 2: '✂️'}

# 每個類別的 BGR 顯示顏色
LABEL_COLOR = {
    0: (60,  60,  220),   # Rock     → 紅
    1: (50,  200,  50),   # Paper    → 綠
    2: (220, 180,   0),   # Scissors → 藍
}

# ── 載入模型 ──────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ 找不到模型：{MODEL_PATH}\n"
        "請先在 train/ 資料夾執行 train_svm.py 產生模型。"
    )

print("⏳ 載入 SVM 模型中...")
clf = joblib.load(MODEL_PATH)
print("✅ 模型載入完成！按 'q' 離開。\n")

# ── 圖片前處理（與訓練時完全一致）────────────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR frame → 正規化一維向量（與 train_svm.py 一致）"""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return resized.flatten() / 255.0

# ── 畫面 UI 輔助函式 ──────────────────────────────────────────────────────────
def draw_ui(frame: np.ndarray, label_idx: int, confidence: float) -> np.ndarray:
    """在畫面上疊加辨識結果與說明文字"""
    h, w = frame.shape[:2]
    color = LABEL_COLOR[label_idx]
    name  = LABEL_NAMES[label_idx]

    # 半透明底色條（頂部）
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 辨識結果文字
    result_text = f"{name}  ({confidence*100:.1f}%)"
    cv2.putText(frame, result_text,
                (16, 48),
                cv2.FONT_HERSHEY_DUPLEX, 1.4,
                color, 2, cv2.LINE_AA)

    # 右下角提示
    cv2.putText(frame, "Press 'q' to quit",
                (w - 220, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)

    # 邊框（顏色對應類別）
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)

    return frame

# ── 主迴圈 ────────────────────────────────────────────────────────────────────
def main():
    # 嘗試開啟攝影機（index 0；若無效請改成 1）
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        # Raspberry Pi 上有時需要指定後端
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("❌ 無法開啟攝影機，請確認連線並嘗試將 index 改為 1。")
        return

    # 設定解析度（可依硬體調整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("📷 攝影機已啟動，開始辨識...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  無法讀取畫面，結束。")
            break

        # 前處理 + 預測
        feature = preprocess(frame).reshape(1, -1)
        label_idx  = int(clf.predict(feature)[0])

        # 取得決策分數作為「信心度」
        try:
            scores     = clf.decision_function(feature)[0]
            # softmax-like 轉換，讓分數落在 0~1
            exp_scores = np.exp(scores - scores.max())
            confidence = float(exp_scores[label_idx] / exp_scores.sum())
        except Exception:
            confidence = 1.0   # SVM 不支援 decision_function 時的後備值

        # 繪製 UI
        frame = draw_ui(frame, label_idx, confidence)

        cv2.imshow("RPS Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 程式結束。")

if __name__ == "__main__":
    main()