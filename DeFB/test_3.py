"""
Blink Detector v2
─────────────────────────────────────────────────────────────────
1단계 : 프레임별 EAR → 4단계 라벨 자동 부착 (Normal/Onset/Valley/Offset)
2단계 : 수집 데이터를 CSV로 저장하고, 경량 DNN으로 실시간 분류

조작키
  Q / ESC : 종료
  S       : 현재까지 수집한 데이터를 CSV로 저장
  T       : CSV가 있으면 DNN 학습 시작 (백그라운드 스레드)
  (학습 완료 후 자동으로 실시간 AI 분류 모드로 전환)
"""

import cv2
import numpy as np
from collections import deque
import time
import os
import csv
import threading

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── 상수 ────────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

CALIBRATION_FRAMES   = 60
SLIDING_WINDOW       = 300
BLINK_CONSEC_FRAMES  = 2
PERCLOS_WINDOW       = 900
PERCLOS_THRESHOLD    = 0.15
FATIGUE_RATIO_THRESH = 0.6
PANEL_W              = 320
SLOPE_THRESH         = 0.005

FEATURE_LEN   = 9          # 입력 피처 수 (아래 make_feature 참조)
SEQUENCE_LEN  = 10         # 시계열 슬라이딩 윈도우 길이
STAGE_LABELS  = ["Normal", "Onset", "Valley", "Offset"]
STAGE_MAP     = {s: i for i, s in enumerate(STAGE_LABELS)}
STAGE_COLORS  = {
    "Normal": (0,   200,  80),
    "Onset":  (0,   165, 255),
    "Valley": (60,   60, 220),
    "Offset": (180, 120,   0),
}

CSV_PATH   = "blink_data.csv"
MODEL_PATH = "blink_model.npz"

# ── MediaPipe 초기화 ─────────────────────────────────────────────
base_options = python.BaseOptions(
    model_asset_path='/Users/hanool/PycharmProjects/PythonProject/DeFB/face_landmarker.task'
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.FaceLandmarker.create_from_options(options)


# ════════════════════════════════════════════════════════════════
#  헬퍼 함수
# ════════════════════════════════════════════════════════════════
def get_3d_ear(landmarks, eye_indices, img_w, img_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w]))
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def get_head_pose_correction(landmarks):
    left  = landmarks[130]
    right = landmarks[359]
    yaw   = abs(left.x - right.x)
    return 1.0 + 0.08 * (1.0 - min(yaw / 0.4, 1.0))


def compute_completeness(ear_open, ear_min):
    if ear_open < 1e-6:
        return 0.0
    return (ear_open - ear_min) / ear_open


def classify_completeness(c):
    if c > 0.9:   return "Complete",   (0, 200, 0)
    elif c > 0.5: return "Incomplete", (0, 165, 255)
    else:         return "Micro",      (0, 0, 255)


def draw_bar(img, x, y, w, h, value, max_val, color_fill, color_bg=(50, 50, 50)):
    cv2.rectangle(img, (x, y), (x+w, y+h), color_bg, -1)
    fill = int(min(value / max(max_val, 1e-6), 1.0) * w)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+h), color_fill, -1)


def txt(img, text, x, y, color=(210, 210, 210), scale=0.50, bold=1):
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)


def divider(img, y):
    cv2.line(img, (14, y), (PANEL_W-14, y), (50, 50, 50), 1)


# ════════════════════════════════════════════════════════════════
#  피처 엔지니어링
# ════════════════════════════════════════════════════════════════
def make_feature(ear_seq, threshold, baseline):
    """
    길이 SEQUENCE_LEN 의 EAR 시퀀스로부터 피처 벡터 생성
    [ear_now, slope_1, slope_2, slope_3,
     ear_norm, slope_norm, ear_min_norm, ear_max_norm, closed_flag]
    """
    seq = list(ear_seq)
    if len(seq) < 2:
        return None
    ear_now  = seq[-1]
    s1 = seq[-1] - seq[-2]
    s2 = (seq[-1] - seq[-3]) / 2.0 if len(seq) >= 3 else s1
    s3 = (seq[-1] - seq[-4]) / 3.0 if len(seq) >= 4 else s2
    ear_norm     = ear_now / max(baseline, 1e-6)
    slope_norm   = s1 / max(baseline, 1e-6)
    ear_min_norm = min(seq[-min(10, len(seq)):]) / max(baseline, 1e-6)
    ear_max_norm = max(seq[-min(10, len(seq)):]) / max(baseline, 1e-6)
    closed_flag  = 1.0 if ear_now < threshold else 0.0
    return np.array([ear_now, s1, s2, s3,
                     ear_norm, slope_norm,
                     ear_min_norm, ear_max_norm,
                     closed_flag], dtype=np.float32)


# ════════════════════════════════════════════════════════════════
#  경량 DNN (NumPy 전용, 외부 라이브러리 불필요)
# ════════════════════════════════════════════════════════════════
class TinyDNN:
    """3층 완전연결망 (9 → 32 → 16 → 4). NumPy만 사용."""

    def __init__(self, in_dim=FEATURE_LEN, hidden1=32, hidden2=16, out_dim=4):
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((in_dim,  hidden1)).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden1, hidden2)).astype(np.float32) * 0.1
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = rng.standard_normal((hidden2, out_dim)).astype(np.float32) * 0.1
        self.b3 = np.zeros(out_dim,  dtype=np.float32)

    # ── 순전파 ──────────────────────────────────────────────────
    def _relu(self, x):    return np.maximum(0, x)
    def _softmax(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, X):
        self.a0 = X
        self.z1 = X @ self.W1 + self.b1;   self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3; self.a3 = self._softmax(self.z3)
        return self.a3

    def predict(self, x):
        return int(np.argmax(self.forward(x[np.newaxis])[0]))

    # ── 역전파 (미니배치 SGD) ────────────────────────────────────
    def _cross_entropy(self, prob, y_oh):
        return -np.mean(np.sum(y_oh * np.log(prob + 1e-9), axis=1))

    def train(self, X, y, epochs=200, lr=0.01, batch=64):
        n_cls = self.W3.shape[1]
        y_oh  = np.eye(n_cls, dtype=np.float32)[y]
        n     = len(X)
        losses = []
        for ep in range(epochs):
            idx = np.random.permutation(n)
            ep_loss = 0.0
            for start in range(0, n, batch):
                bx = X[idx[start:start+batch]]
                by = y_oh[idx[start:start+batch]]
                prob = self.forward(bx)
                loss = self._cross_entropy(prob, by)
                ep_loss += loss

                # ── 역전파 ──────────────────────────────────────
                dz3 = (prob - by) / len(bx)
                dW3 = self.a2.T @ dz3;  db3 = dz3.sum(0)
                da2 = dz3 @ self.W3.T
                dz2 = da2 * (self.z2 > 0)
                dW2 = self.a1.T @ dz2;  db2 = dz2.sum(0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (self.z1 > 0)
                dW1 = self.a0.T @ dz1;  db1 = dz1.sum(0)

                self.W3 -= lr * dW3; self.b3 -= lr * db3
                self.W2 -= lr * dW2; self.b2 -= lr * db2
                self.W1 -= lr * dW1; self.b1 -= lr * db1

            losses.append(ep_loss)
            if (ep + 1) % 50 == 0:
                acc = np.mean(np.argmax(self.forward(X), axis=1) == y)
                print(f"  epoch {ep+1:>3}/{epochs}  loss={ep_loss:.4f}  acc={acc*100:.1f}%")
        return losses

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        print(f"Model saved → {path}")

    def load(self, path):
        d = np.load(path)
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
        self.W3, self.b3 = d["W3"], d["b3"]
        print(f"Model loaded ← {path}")


# ════════════════════════════════════════════════════════════════
#  데이터 수집 / 학습 유틸
# ════════════════════════════════════════════════════════════════
collected_rows = []   # (feature_vec, rule_label_idx) 를 누적

def save_csv(rows, path=CSV_PATH):
    header = ["ear", "s1", "s2", "s3",
              "ear_norm", "slope_norm", "ear_min_norm", "ear_max_norm",
              "closed_flag", "label"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for feat, lbl in rows:
            w.writerow(list(feat) + [lbl])
    print(f"CSV saved ({len(rows)} rows) → {path}")


def load_csv(path=CSV_PATH):
    X, y = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[k]) for k in
                      ["ear", "s1", "s2", "s3",
                       "ear_norm", "slope_norm",
                       "ear_min_norm", "ear_max_norm", "closed_flag"]])
            y.append(int(row["label"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── 학습 스레드 ──────────────────────────────────────────────────
model        = TinyDNN()
model_ready  = False
training_msg = ""

def train_thread_fn():
    global model, model_ready, training_msg
    training_msg = "Training..."
    try:
        X, y = load_csv()
        # 클래스별 오버샘플링 (불균형 보정)
        counts = np.bincount(y, minlength=4)
        max_c  = counts.max()
        X_list, y_list = [X], [y]
        for cls in range(4):
            idx = np.where(y == cls)[0]
            if len(idx) == 0: continue
            extra = max_c - len(idx)
            if extra > 0:
                rep = idx[np.random.randint(0, len(idx), extra)]
                X_list.append(X[rep]); y_list.append(y[rep])
        X_bal = np.concatenate(X_list); y_bal = np.concatenate(y_list)
        perm  = np.random.permutation(len(X_bal))
        X_bal, y_bal = X_bal[perm], y_bal[perm]

        model = TinyDNN()
        model.train(X_bal, y_bal, epochs=300, lr=0.005, batch=128)
        model.save(MODEL_PATH)
        model_ready  = True
        training_msg = f"AI ready! ({len(X)} samples)"
    except Exception as e:
        training_msg = f"Train error: {e}"
        print(training_msg)


# ════════════════════════════════════════════════════════════════
#  메인 상태
# ════════════════════════════════════════════════════════════════
calib_ears         = []
personal_threshold = None
ear_open_baseline  = None
calibrated         = False
ear_window         = deque(maxlen=SLIDING_WINDOW)
ear_seq            = deque(maxlen=SEQUENCE_LEN)   # 피처용 짧은 시퀀스
eye_closed_log     = deque(maxlen=PERCLOS_WINDOW)
closing_slopes     = deque(maxlen=20)
opening_slopes     = deque(maxlen=20)

blink_total       = 0
incomplete_total  = 0
micro_total       = 0
blink_consec      = 0
in_blink          = False
blink_ear_min     = 1.0
prev_ear          = None
start_time        = time.time()

blink_stage       = "Normal"
ai_stage          = "Normal"
stage_history     = deque(maxlen=SLIDING_WINDOW)
stage_frame_count = {s: 0 for s in STAGE_LABELS}

# 저장된 모델이 있으면 미리 로드
if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH)
    model_ready  = True
    training_msg = f"Model pre-loaded"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Started.  S=Save CSV  T=Train  Q/ESC=Quit")

# ════════════════════════════════════════════════════════════════
#  메인 루프
# ════════════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    fh, fw  = frame.shape[:2]
    elapsed = time.time() - start_time
    overlay = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mpi)

    if res.face_landmarks:
        lms = res.face_landmarks[0]

        le  = get_3d_ear(lms, LEFT_EYE,  fw, fh)
        re  = get_3d_ear(lms, RIGHT_EYE, fw, fh)
        cor = get_head_pose_correction(lms)
        ear = ((le + re) / 2.0) * cor
        ear_window.append(ear)
        ear_seq.append(ear)

        # ── Calibration ──────────────────────────────────────────
        if not calibrated:
            calib_ears.append(ear)
            prog = len(calib_ears) / CALIBRATION_FRAMES
            bw   = int(480 * prog)
            cx, cy = fw//2, fh//2
            cv2.rectangle(overlay, (cx-240, cy-40), (cx+240, cy+40), (30, 30, 30), -1)
            cv2.rectangle(overlay, (cx-220, cy-14), (cx+220, cy+14), (55, 55, 55), -1)
            if bw > 0:
                cv2.rectangle(overlay, (cx-220, cy-14), (cx-220+bw, cy+14), (0, 180, 90), -1)
            cv2.putText(overlay, f"Calibrating... {int(prog*100)}%",
                        (cx-130, cy-22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (200, 200, 200), 2, cv2.LINE_AA)
            if len(calib_ears) >= CALIBRATION_FRAMES:
                ear_open_baseline  = np.percentile(calib_ears, 80)
                personal_threshold = ear_open_baseline * 0.75
                calibrated         = True
                print(f"Calibration done  threshold={personal_threshold:.3f}")
            cv2.imshow("Blink Detector", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            continue

        # ── Drift correction ──────────────────────────────────────
        if len(ear_window) >= 30:
            ro = np.percentile(list(ear_window), 80)
            if ro > ear_open_baseline * 0.5:
                personal_threshold = ro * 0.75

        # ── Slope ─────────────────────────────────────────────────
        slope    = (ear - prev_ear) if prev_ear is not None else 0.0
        is_closed = ear < personal_threshold

        # ── 1단계: 규칙 기반 Stage 라벨 ─────────────────────────
        if not is_closed:
            blink_stage = "Normal"
        else:
            if slope < -SLOPE_THRESH:   blink_stage = "Onset"
            elif slope > SLOPE_THRESH:  blink_stage = "Offset"
            else:                       blink_stage = "Valley"

        stage_history.append(blink_stage)
        stage_frame_count[blink_stage] += 1

        # ── 2단계: 피처 추출 & 데이터 수집 ──────────────────────
        feat = make_feature(ear_seq, personal_threshold, ear_open_baseline)
        if feat is not None:
            rule_lbl = STAGE_MAP[blink_stage]
            collected_rows.append((feat, rule_lbl))   # 수집

            # AI 추론 (모델이 준비됐을 때)
            if model_ready:
                ai_lbl   = model.predict(feat)
                ai_stage = STAGE_LABELS[ai_lbl]

        # ── Slope logging ─────────────────────────────────────────
        if slope < -SLOPE_THRESH: closing_slopes.append(abs(slope))
        elif slope > SLOPE_THRESH: opening_slopes.append(abs(slope))

        # ── Blink logic ───────────────────────────────────────────
        if is_closed:
            blink_consec  += 1
            blink_ear_min  = min(blink_ear_min, ear)
            in_blink       = True
            eye_closed_log.append(1)
        else:
            if in_blink and blink_consec >= BLINK_CONSEC_FRAMES:
                blink_total += 1
                c = compute_completeness(ear_open_baseline, blink_ear_min)
                label, _ = classify_completeness(c)
                if label == "Incomplete": incomplete_total += 1
                elif label == "Micro":    micro_total      += 1
            in_blink      = False
            blink_ear_min = 1.0
            blink_consec  = 0
            eye_closed_log.append(0)

        prev_ear = ear

        # ── Metrics ───────────────────────────────────────────────
        perclos       = sum(eye_closed_log) / max(len(eye_closed_log), 1)
        avg_c         = np.mean(closing_slopes) if closing_slopes else 0.01
        avg_o         = np.mean(opening_slopes) if opening_slopes else 0.01
        fatigue_ratio = avg_o / (avg_c + 1e-6)
        bpm           = blink_total / max(elapsed / 60.0, 1/60.0)
        inc_ratio     = incomplete_total / max(blink_total, 1)

        # Eye dots
        for idx in LEFT_EYE + RIGHT_EYE:
            lm = lms[idx]
            cv2.circle(overlay, (int(lm.x*fw), int(lm.y*fh)), 2, (0, 230, 180), -1)

        # ════════════════════════════════════════════════════════
        #  패널 그리기
        # ════════════════════════════════════════════════════════
        panel = np.full((fh, PANEL_W, 3), (18, 18, 18), dtype=np.uint8)

        # Title
        txt(panel, "BLINK MONITOR", 14, 34, (90, 200, 255), 0.62, 2)
        txt(panel, f"{int(elapsed//60):02d}:{int(elapsed%60):02d}",
            14, 54, (90, 90, 90), 0.42)

        # ── EAR ──────────────────────────────────────────────────
        divider(panel, 66)
        txt(panel, "EAR", 14, 84, (110, 110, 110), 0.42)
        txt(panel, f"Current    {ear:.3f}", 14, 106)
        txt(panel, f"Threshold  {personal_threshold:.3f}", 14, 126, (150, 150, 150), 0.46)
        ear_col = (0, 200, 80) if not is_closed else (60, 60, 220)
        draw_bar(panel, 14, 136, PANEL_W-28, 10, ear, ear_open_baseline, ear_col)

        # ── Blink Stats ───────────────────────────────────────────
        divider(panel, 158)
        txt(panel, "BLINK STATS", 14, 176, (110, 110, 110), 0.42)
        txt(panel, f"Total   {blink_total}", 14, 200)
        bpm_col = (0, 200, 80) if 12 <= bpm <= 20 else (60, 100, 230)
        txt(panel, f"BPM     {bpm:.1f}", 14, 222, bpm_col)
        inc_col = (0, 165, 255) if inc_ratio > 0.3 else (160, 160, 160)
        txt(panel, f"Incomplete  {incomplete_total}  ({inc_ratio*100:.0f}%)", 14, 244, inc_col)
        txt(panel, f"Micro   {micro_total}", 14, 264, (140, 140, 140))

        # ── PERCLOS ───────────────────────────────────────────────
        divider(panel, 280)
        txt(panel, "PERCLOS", 14, 298, (110, 110, 110), 0.42)
        pc_col = (60, 60, 220) if perclos > PERCLOS_THRESHOLD else (0, 200, 80)
        txt(panel, f"{perclos*100:.1f}%", 14, 322, pc_col, 0.62, 2)
        txt(panel, f"alert > {PERCLOS_THRESHOLD*100:.0f}%", 14, 342, (90, 90, 90), 0.42)
        draw_bar(panel, 14, 352, PANEL_W-28, 10, perclos, 0.30, pc_col)
        tx = 14 + int(PERCLOS_THRESHOLD / 0.30 * (PANEL_W-28))
        cv2.line(panel, (tx, 348), (tx, 366), (180, 60, 60), 2)

        # ── Fatigue ───────────────────────────────────────────────
        divider(panel, 378)
        txt(panel, "FATIGUE", 14, 396, (110, 110, 110), 0.42)
        fr_col = (60, 60, 220) if fatigue_ratio < FATIGUE_RATIO_THRESH else (0, 200, 80)
        txt(panel, f"Open/Close  {fatigue_ratio:.2f}", 14, 420, fr_col)
        txt(panel, f"alert < {FATIGUE_RATIO_THRESH}", 14, 440, (90, 90, 90), 0.42)

        # ── Status ────────────────────────────────────────────────
        divider(panel, 456)
        txt(panel, "STATUS", 14, 474, (110, 110, 110), 0.42)
        if is_closed:
            cv2.rectangle(panel, (14, 484), (PANEL_W-14, 514), (35, 15, 15), -1)
            cv2.rectangle(panel, (14, 484), (18, 514), (60, 60, 220), -1)
            txt(panel, "EYE CLOSED", 28, 506, (80, 80, 230), 0.60, 2)
        else:
            cv2.rectangle(panel, (14, 484), (PANEL_W-14, 514), (15, 35, 15), -1)
            cv2.rectangle(panel, (14, 484), (18, 514), (0, 200, 80), -1)
            txt(panel, "EYE OPEN", 28, 506, (0, 200, 80), 0.60, 2)

        # ── 1단계: 규칙 기반 Stage ────────────────────────────────
        divider(panel, 530)
        txt(panel, "BLINK STAGE  [Rule]", 14, 548, (110, 110, 110), 0.42)
        sc = STAGE_COLORS[blink_stage]
        cv2.rectangle(panel, (14, 556), (PANEL_W-14, 584), (30, 30, 30), -1)
        cv2.rectangle(panel, (14, 556), (18, 584), sc, -1)
        txt(panel, blink_stage.upper(), 28, 578, sc, 0.62, 2)

        # 단계별 누적 프레임
        for i, (sname, scolor) in enumerate(STAGE_COLORS.items()):
            txt(panel, f"{sname:<8} {stage_frame_count[sname]:>5}f",
                14, 596 + i * 18, scolor, 0.38)

        # ── 2단계: AI Stage ───────────────────────────────────────
        divider(panel, 672)
        txt(panel, "BLINK STAGE  [AI]", 14, 690, (110, 110, 110), 0.42)
        if model_ready:
            asc = STAGE_COLORS[ai_stage]
            cv2.rectangle(panel, (14, 698), (PANEL_W-14, 726), (30, 30, 30), -1)
            cv2.rectangle(panel, (14, 698), (18, 726), asc, -1)
            txt(panel, ai_stage.upper(), 28, 720, asc, 0.62, 2)
        else:
            txt(panel, training_msg if training_msg else "Press T to train",
                14, 716, (100, 100, 100), 0.42)

        # 수집 데이터 수
        txt(panel, f"Collected: {len(collected_rows)} rows  (S=save T=train)",
            14, fh - 16, (70, 70, 70), 0.38)

        overlay[:, :PANEL_W] = panel

        # ── Warnings ─────────────────────────────────────────────
        warnings = []
        if perclos > PERCLOS_THRESHOLD:
            warnings.append(("PERCLOS alert: blink more!", (60, 60, 210)))
        if fatigue_ratio < FATIGUE_RATIO_THRESH and blink_total > 5:
            warnings.append(("Eye fatigue detected", (60, 100, 220)))
        if bpm < 8 and elapsed > 30:
            warnings.append(("Low blink rate!", (60, 140, 210)))
        if inc_ratio > 0.5 and blink_total > 5:
            warnings.append(("Too many incomplete blinks", (60, 120, 210)))

        for i, (msg, col) in enumerate(warnings):
            yw = fh - 16 - i * 34
            cv2.rectangle(overlay, (PANEL_W+10, yw-22), (fw-10, yw+10), (28, 28, 28), -1)
            cv2.rectangle(overlay, (PANEL_W+10, yw-22), (PANEL_W+15, yw+10), col, -1)
            cv2.putText(overlay, msg, (PANEL_W+22, yw),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)

    else:
        overlay[:, :PANEL_W] = (18, 18, 18)
        cv2.putText(overlay, "No face detected",
                    (PANEL_W + (fw-PANEL_W)//2 - 140, fh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 180), 2, cv2.LINE_AA)

    cv2.imshow("Blink Detector", overlay)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), 27):
        break
    elif key == ord('s') and collected_rows:
        save_csv(collected_rows)
    elif key == ord('t'):
        if os.path.exists(CSV_PATH):
            t = threading.Thread(target=train_thread_fn, daemon=True)
            t.start()
        else:
            print("No CSV found. Press S first to save data.")

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Done.")