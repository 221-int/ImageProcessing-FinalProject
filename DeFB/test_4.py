"""
Fatigue Detector v4
─────────────────────────────────────────────────────────────────
v3에서 추가된 기능:
  [A] MAR (Mouth Aspect Ratio) - 하품 감지
  [B] Head Pose (Pitch/Yaw/Roll) - 고개 숙임 감지
  [C] 종합 피로도 인덱스 (Fatigue Index) - 5분 단위 시계열 로그
  [D] 키 입력을 별도 스레드로 처리 (macOS OpenCV 포커스 문제 해결)

조작키 (터미널 또는 PyCharm 콘솔에서 입력 후 엔터)
  S : 깜빡임 데이터 CSV 저장
  T : DNN 학습 시작 (백그라운드)
  L : 피로도 로그 CSV 저장
  Q : 종료
"""

import cv2
import numpy as np
from collections import deque
import time
import os
import csv
import threading
import math
import sys
import tty
import termios
import select

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ════════════════════════════════════════════════════════════════
#  상수 정의
# ════════════════════════════════════════════════════════════════

# MediaPipe 얼굴 랜드마크에서 눈과 입에 해당하는 인덱스 번호
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH_IDX = [61, 291, 13, 14, 78, 308]

# Head Pose 계산에 사용할 랜드마크 인덱스
NOSE_TIP    = 1
CHIN        = 152
LEFT_EYE_L  = 33
RIGHT_EYE_R = 263
LEFT_MOUTH  = 61
RIGHT_MOUTH = 291

CALIBRATION_FRAMES   = 60     # 처음 60프레임 동안 개인 눈 크기 캘리브레이션
SLIDING_WINDOW       = 300    # 최근 300프레임 기준으로 EAR 추세 보정
BLINK_CONSEC_FRAMES  = 2      # 연속 2프레임 이상 눈 감김 → 깜빡임으로 인정
PERCLOS_WINDOW       = 900    # PERCLOS 계산 구간 (약 30초)
PERCLOS_THRESHOLD    = 0.15   # 15% 이상 눈 감기면 피로 경고
FATIGUE_RATIO_THRESH = 0.6    # 눈 뜨는 속도 / 감는 속도 비율 임계값
PANEL_W              = 340    # 왼쪽 정보 패널 너비
SLOPE_THRESH         = 0.005  # EAR 기울기 임계값 (깜빡임 단계 구분에 사용)

MAR_YAWN_THRESH   = 0.6   # MAR이 이 값 초과하면 하품으로 판단
MAR_CONSEC_FRAMES = 15    # 하품으로 인정하기 위한 최소 지속 프레임 수
PITCH_THRESH      = 15.0  # 고개 숙임 각도 임계값 (도)
YAW_THRESH        = 25.0  # 고개 돌림 각도 임계값 (도)
LOG_INTERVAL_SEC  = 300   # 5분마다 피로도 로그 저장

# 종합 피로도 인덱스 계산 시 각 지표별 가중치
W_PERCLOS   = 0.35   # PERCLOS 가중치 (가장 중요)
W_INC_BLINK = 0.25   # 불완전 깜빡임 비율 가중치
W_YAWN      = 0.20   # 하품 빈도 가중치
W_HEAD      = 0.20   # 고개 숙임 비율 가중치

SEQUENCE_LEN = 10    # 피처 생성에 사용할 EAR 시퀀스 길이
STAGE_LABELS = ["Normal", "Onset", "Valley", "Offset"]  # 깜빡임 4단계
STAGE_MAP    = {s: i for i, s in enumerate(STAGE_LABELS)}
STAGE_COLORS = {
    "Normal": (0,   200,  80),
    "Onset":  (0,   165, 255),
    "Valley": (60,   60, 220),
    "Offset": (180, 120,   0),
}
FEATURE_LEN = SEQUENCE_LEN + 4  # 총 피처 수: EAR 시퀀스 10개 + slope 3개 + closed_flag 1개

CSV_PATH    = "blink_data.csv"   # 수집 데이터 저장 경로
MODEL_PATH  = "blink_model.npz"  # 학습된 모델 저장 경로
FATIGUE_LOG = "fatigue_log.csv"  # 피로도 로그 저장 경로


# ════════════════════════════════════════════════════════════════
#  키 입력 스레드 (macOS에서 OpenCV 창이 포커스를 못 받는 문제 우회)
# ════════════════════════════════════════════════════════════════
_key_queue  = deque(maxlen=10)  # 입력된 키를 임시로 쌓아두는 큐
_stop_input = False              # 스레드 종료 신호

_is_tty = sys.stdin.isatty()    # 진짜 터미널인지 PyCharm 콘솔인지 판별

def _key_listener():
    if _is_tty:
        # 터미널 환경: raw 모드로 즉시 한 글자씩 읽기
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not _stop_input:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    _key_queue.append(ch.lower())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    else:
        # PyCharm 등 비-TTY 환경: 한 줄씩 읽기 (엔터 후 처리)
        print("PyCharm 모드: 명령 입력 후 엔터 (s=저장 t=학습 l=로그 q=종료)")
        while not _stop_input:
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline().strip().lower()
                    for ch in line:
                        _key_queue.append(ch)
            except (EOFError, OSError):
                break

_input_thread = threading.Thread(target=_key_listener, daemon=True)
_input_thread.start()


# ── MediaPipe 얼굴 랜드마크 감지기 초기화 ────────────────────────
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
#  MAR (Mouth Aspect Ratio) 계산
#  EAR과 같은 원리로, 입의 세로/가로 비율로 하품 여부를 판단
# ════════════════════════════════════════════════════════════════
def get_mar(landmarks, img_w, img_h):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h])
    left,  right = pt(MOUTH_IDX[0]), pt(MOUTH_IDX[1])
    top1,  bot1  = pt(MOUTH_IDX[2]), pt(MOUTH_IDX[3])
    top2,  bot2  = pt(MOUTH_IDX[4]), pt(MOUTH_IDX[5])
    vert1 = np.linalg.norm(top1 - bot1)
    vert2 = np.linalg.norm(top2 - bot2)
    horiz = np.linalg.norm(left  - right)
    return (vert1 + vert2) / (2.0 * horiz + 1e-6)


# ════════════════════════════════════════════════════════════════
#  Head Pose 추정
#  solvePnP로 2D 랜드마크 → 3D 회전각(Pitch/Yaw/Roll) 변환
# ════════════════════════════════════════════════════════════════

# 표준 3D 얼굴 모델 좌표 (코 끝, 턱, 양쪽 눈·입꼬리)
FACE_3D = np.array([
    [ 0.0,    0.0,    0.0  ],
    [ 0.0,  -63.6,  -12.5 ],
    [-43.3,  32.7,  -26.0 ],
    [ 43.3,  32.7,  -26.0 ],
    [-28.9, -28.9,  -24.1 ],
    [ 28.9, -28.9,  -24.1 ],
], dtype=np.float64)
POSE_LM_IDX = [NOSE_TIP, CHIN, LEFT_EYE_L, RIGHT_EYE_R, LEFT_MOUTH, RIGHT_MOUTH]

def get_head_pose(landmarks, img_w, img_h):
    # 카메라에서 감지된 2D 좌표 추출
    face_2d = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in POSE_LM_IDX],
        dtype=np.float64
    )
    focal   = img_w
    cam_mat = np.array([[focal,0,img_w/2],[0,focal,img_h/2],[0,0,1]], dtype=np.float64)
    dist    = np.zeros((4,1), dtype=np.float64)
    # 3D 모델과 2D 좌표를 맞춰서 회전 벡터 계산
    ok, rvec, _ = cv2.solvePnP(FACE_3D, face_2d, cam_mat, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)  # 회전 벡터 → 회전 행렬
    sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2( rmat[2,1], rmat[2,2]))  # 고개 숙임
        yaw   = math.degrees(math.atan2(-rmat[2,0], sy))          # 고개 좌우 회전
        roll  = math.degrees(math.atan2( rmat[1,0], rmat[0,0]))   # 고개 기울임
    else:
        pitch = math.degrees(math.atan2(-rmat[1,2], rmat[1,1]))
        yaw   = math.degrees(math.atan2(-rmat[2,0], sy))
        roll  = 0.0
    return pitch, yaw, roll


# ════════════════════════════════════════════════════════════════
#  종합 피로도 인덱스 계산
#  PERCLOS, 불완전 깜빡임, 하품, 고개 숙임을 가중 합산
# ════════════════════════════════════════════════════════════════
def compute_fatigue_index(perclos, inc_ratio, yawn_rpm, head_droop_ratio):
    yawn_norm = min(yawn_rpm / 3.0, 1.0)  # 하품 빈도를 0~1로 정규화
    fi = (W_PERCLOS   * min(perclos / PERCLOS_THRESHOLD, 1.0) +
          W_INC_BLINK * inc_ratio +
          W_YAWN      * yawn_norm +
          W_HEAD      * head_droop_ratio)
    return min(fi, 1.0)

def fatigue_level(fi):
    # 피로도 수치를 3단계 등급으로 변환
    if fi < 0.3:   return "NORMAL",  (0,   200,  80)
    elif fi < 0.6: return "CAUTION", (0,   165, 255)
    else:          return "DANGER",  (60,   60, 220)


# ════════════════════════════════════════════════════════════════
#  EAR 및 기타 함수
# ════════════════════════════════════════════════════════════════
def get_3d_ear(landmarks, eye_indices, img_w, img_h):
    # 눈 랜드마크 6개로 EAR(Eye Aspect Ratio) 계산
    # EAR = (세로 거리 합) / (가로 거리 * 2)
    pts = [np.array([landmarks[i].x*img_w, landmarks[i].y*img_h, landmarks[i].z*img_w])
           for i in eye_indices]
    v1 = np.linalg.norm(pts[1]-pts[5])
    v2 = np.linalg.norm(pts[2]-pts[4])
    h  = np.linalg.norm(pts[0]-pts[3])
    return (v1+v2)/(2.0*h+1e-6)

def get_head_pose_correction(landmarks):
    # 고개가 옆을 향할수록 눈이 작게 보이므로 EAR 보정
    yaw = abs(landmarks[130].x - landmarks[359].x)
    return 1.0 + 0.08*(1.0 - min(yaw/0.4, 1.0))

def compute_completeness(ear_open, ear_min):
    # 깜빡임 완전성: 1에 가까울수록 완전한 깜빡임
    return 0.0 if ear_open < 1e-6 else (ear_open-ear_min)/ear_open

def classify_completeness(c):
    # 완전성 수치로 깜빡임 종류 분류
    if c > 0.9:   return "Complete",   (0,200,0)
    elif c > 0.5: return "Incomplete", (0,165,255)
    else:         return "Micro",      (0,0,255)

def draw_bar(img, x, y, w, h, value, max_val, color_fill, color_bg=(50,50,50)):
    cv2.rectangle(img, (x,y), (x+w,y+h), color_bg, -1)
    fill = int(min(value/max(max_val,1e-6),1.0)*w)
    if fill > 0:
        cv2.rectangle(img, (x,y), (x+fill,y+h), color_fill, -1)

def txt(img, text, x, y, color=(210,210,210), scale=0.50, bold=1):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

def divider(img, y):
    cv2.line(img, (14,y), (PANEL_W-14,y), (50,50,50), 1)

def make_feature(ear_seq, threshold, baseline):
    # AI 입력 피처 벡터 생성
    # f0~f9: 최근 10프레임 EAR을 baseline으로 정규화한 시퀀스
    # f10~f12: EAR 변화 기울기 (1/2/3프레임 단위)
    # f13: 현재 눈 감김 여부 (0 또는 1)
    seq = list(ear_seq)
    if len(seq) < SEQUENCE_LEN:
        seq = [seq[0]]*(SEQUENCE_LEN-len(seq)) + seq  # 짧으면 앞을 첫 값으로 채움
    norm_seq    = [e/max(baseline,1e-6) for e in seq]
    s1 = seq[-1]-seq[-2]
    s2 = (seq[-1]-seq[-3])/2.0
    s3 = (seq[-1]-seq[-4])/3.0
    closed_flag = 1.0 if seq[-1] < threshold else 0.0
    return np.array(norm_seq+[s1/max(baseline,1e-6),
                               s2/max(baseline,1e-6),
                               s3/max(baseline,1e-6),
                               closed_flag], dtype=np.float32)


# ════════════════════════════════════════════════════════════════
#  TinyDNN - NumPy만으로 구현한 경량 3층 신경망
#  구조: 입력(14) → 은닉(64) → 은닉(32) → 출력(4)
# ════════════════════════════════════════════════════════════════
class TinyDNN:
    def __init__(self, in_dim=FEATURE_LEN, h1=64, h2=32, out=4):
        # He 초기화: ReLU 사용 시 권장되는 가중치 초기화 방법
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((in_dim,h1)).astype(np.float32)*np.sqrt(2/in_dim)
        self.b1 = np.zeros(h1,  dtype=np.float32)
        self.W2 = rng.standard_normal((h1,h2)).astype(np.float32)*np.sqrt(2/h1)
        self.b2 = np.zeros(h2,  dtype=np.float32)
        self.W3 = rng.standard_normal((h2,out)).astype(np.float32)*np.sqrt(2/h2)
        self.b3 = np.zeros(out, dtype=np.float32)

    def _relu(self,x): return np.maximum(0,x)
    def _softmax(self,x):
        e = np.exp(x-x.max(axis=-1,keepdims=True))
        return e/e.sum(axis=-1,keepdims=True)

    def forward(self,X):
        # 순전파: 입력 → 은닉층1 → 은닉층2 → 출력(확률)
        self.a0=X
        self.z1=X@self.W1+self.b1;    self.a1=self._relu(self.z1)
        self.z2=self.a1@self.W2+self.b2; self.a2=self._relu(self.z2)
        self.z3=self.a2@self.W3+self.b3; self.a3=self._softmax(self.z3)
        return self.a3

    def predict(self,x):
        # 단일 샘플 예측: 가장 높은 확률의 클래스 인덱스 반환
        return int(np.argmax(self.forward(x[np.newaxis])[0]))

    def _ce(self,p,y): return -np.mean(np.sum(y*np.log(p+1e-9),axis=1))

    def train(self,X,y,epochs=300,lr=0.005,batch=128):
        nc   = self.W3.shape[1]
        n    = len(X)
        sp   = int(n*0.8)
        pm   = np.random.permutation(n)
        Xtr,ytr   = X[pm[:sp]], y[pm[:sp]]    # 학습 80%
        Xval,yval = X[pm[sp:]], y[pm[sp:]]    # 검증 20%
        yoh = np.eye(nc,dtype=np.float32)[ytr]
        bv,bw = 0.0, self._gw()
        for ep in range(epochs):
            idx=np.random.permutation(len(Xtr)); el=0.0
            for s in range(0,len(Xtr),batch):
                bx=Xtr[idx[s:s+batch]]; by=yoh[idx[s:s+batch]]
                p=self.forward(bx); el+=self._ce(p,by)
                # 역전파: 각 레이어 가중치 기울기 계산
                dz3=(p-by)/len(bx)
                dW3=self.a2.T@dz3; db3=dz3.sum(0)
                dz2=(dz3@self.W3.T)*(self.z2>0)
                dW2=self.a1.T@dz2; db2=dz2.sum(0)
                dz1=(dz2@self.W2.T)*(self.z1>0)
                dW1=self.a0.T@dz1; db1=dz1.sum(0)
                # SGD로 가중치 업데이트
                self.W3-=lr*dW3; self.b3-=lr*db3
                self.W2-=lr*dW2; self.b2-=lr*db2
                self.W1-=lr*dW1; self.b1-=lr*db1
            if (ep+1)%50==0:
                ta=np.mean(np.argmax(self.forward(Xtr), axis=1)==ytr)
                va=np.mean(np.argmax(self.forward(Xval),axis=1)==yval)
                print(f"  ep{ep+1}/{epochs} loss={el:.3f} tr={ta*100:.1f}% val={va*100:.1f}%")
                if va>bv: bv=va; bw=self._gw()  # 검증 정확도 최고일 때 가중치 저장
        self._sw(bw); print(f"Best val={bv*100:.1f}%")

    def _gw(self): return tuple(w.copy() for w in [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3])
    def _sw(self,w): self.W1,self.b1,self.W2,self.b2,self.W3,self.b3=[x.copy() for x in w]

    def save(self,path):
        np.savez(path,W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,W3=self.W3,b3=self.b3)
        print(f"Model saved → {path}")

    def load(self,path):
        d=np.load(path)
        self.W1,self.b1=d["W1"],d["b1"]
        self.W2,self.b2=d["W2"],d["b2"]
        self.W3,self.b3=d["W3"],d["b3"]
        print(f"Model loaded ← {path}")


# ════════════════════════════════════════════════════════════════
#  데이터 저장 / 불러오기 / 학습 유틸
# ════════════════════════════════════════════════════════════════
collected_rows = []  # 실시간으로 수집된 (피처, 라벨) 쌍을 누적
fatigue_log    = []  # 5분 단위 피로도 스냅샷 로그

def save_csv(rows, path=CSV_PATH):
    cols = [f"f{i}" for i in range(FEATURE_LEN)]
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(cols+["label"])
        for feat,lbl in rows: w.writerow(list(feat)+[lbl])
    print(f"CSV saved ({len(rows)} rows) → {path}")

def load_csv(path=CSV_PATH):
    X,y=[],[]
    cols=[f"f{i}" for i in range(FEATURE_LEN)]
    with open(path,newline="") as f:
        for row in csv.DictReader(f):
            X.append([float(row[k]) for k in cols]); y.append(int(row["label"]))
    return np.array(X,np.float32), np.array(y,np.int32)

def balance(X,y):
    # 클래스 불균형 보정: Normal이 압도적으로 많아서 비율 맞춰 샘플링
    c=np.bincount(y,minlength=4)
    nm=max(c[1],c[2],c[3])
    if nm < 5:
        # 깜빡임 데이터가 너무 적으면 그냥 원본으로 학습
        print(f"  [경고] Onset/Valley/Offset 샘플 부족 ({c[1]}/{c[2]}/{c[3]}개). 원본 데이터로 학습합니다.")
        p=np.random.permutation(len(X))
        return X[p],y[p]
    nc=min(c[0],nm*3); tg=max(nm,100)
    Xo,yo=[],[]
    for cls in range(4):
        idx=np.where(y==cls)[0]
        if not len(idx): continue
        cap=nc if cls==0 else tg
        ch=(np.random.choice(idx,cap,replace=False) if len(idx)>=cap
            else np.concatenate([idx,np.random.choice(idx,cap-len(idx),replace=True)]))
        Xo.append(X[ch]); yo.append(y[ch])
    Xb=np.concatenate(Xo); yb=np.concatenate(yo)
    p=np.random.permutation(len(Xb))
    return Xb[p],yb[p]

def save_fatigue_log():
    with open(FATIGUE_LOG,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["time_sec","fatigue_index","perclos","inc_ratio","yawn_rpm","head_droop"])
        for row in fatigue_log: w.writerow(row)
    print(f"Fatigue log saved ({len(fatigue_log)} entries) → {FATIGUE_LOG}")

_model_lock  = threading.Lock()  # 멀티스레드에서 모델 동시 접근 방지
model        = TinyDNN()
model_ready  = False
training_msg = ""

def train_thread_fn():
    # 학습은 메인 루프를 막지 않도록 별도 스레드에서 실행
    global model, model_ready, training_msg
    training_msg = "Training..."
    try:
        X,y = load_csv()
        Xb,yb = balance(X,y)
        print(f"Balanced {len(Xb)} rows  cls={np.bincount(yb,minlength=4)}")
        nm = TinyDNN(); nm.train(Xb,yb); nm.save(MODEL_PATH)
        with _model_lock:
            model=nm; model_ready=True
            training_msg=f"AI ready! ({len(X)} samples)"
    except Exception as e:
        training_msg=f"Train error: {e}"; print(training_msg)

# 이전에 저장된 모델이 있으면 자동으로 불러옴 (피처 크기도 검증)
if os.path.exists(MODEL_PATH):
    try:
        d = np.load(MODEL_PATH)
        if d["W1"].shape[0] == FEATURE_LEN:
            model.load(MODEL_PATH)
            model_ready  = True
            training_msg = "Model pre-loaded"
        else:
            print(f"Model dim mismatch ({d['W1'].shape[0]} vs {FEATURE_LEN}). Skipping.")
    except Exception as e:
        print(f"Model load failed: {e}")


# ════════════════════════════════════════════════════════════════
#  메인 상태 변수 초기화
# ════════════════════════════════════════════════════════════════
calib_ears         = []                         # 캘리브레이션용 EAR 수집 버퍼
personal_threshold = None                       # 개인별 눈 감김 임계값
ear_open_baseline  = None                       # 눈 떴을 때 EAR 기준값
calibrated         = False
ear_window         = deque(maxlen=SLIDING_WINDOW)   # EAR 추세 보정용
ear_seq            = deque(maxlen=SEQUENCE_LEN)     # 피처 생성용 시퀀스
eye_closed_log     = deque(maxlen=PERCLOS_WINDOW)   # PERCLOS 계산용
closing_slopes     = deque(maxlen=20)               # 감는 속도 기록
opening_slopes     = deque(maxlen=20)               # 뜨는 속도 기록

blink_total      = 0      # 총 깜빡임 횟수
incomplete_total = 0      # 불완전 깜빡임 횟수
micro_total      = 0      # 마이크로 깜빡임 횟수
blink_consec     = 0      # 현재 연속으로 눈 감긴 프레임 수
in_blink         = False  # 깜빡임 진행 중 여부
blink_ear_min    = 1.0    # 현재 깜빡임에서 EAR 최저값
prev_ear         = None
start_time       = time.time()
last_log_time    = time.time()

blink_stage       = "Normal"
ai_stage          = "Normal"
_blink_phase      = "none"   # 상태 머신 현재 단계: none / onset / valley / offset
stage_history     = deque(maxlen=SLIDING_WINDOW)
stage_frame_count = {s:0 for s in STAGE_LABELS}

mar              = 0.0
yawn_consec      = 0
yawn_total       = 0
in_yawn          = False

pitch            = 0.0
yaw_angle        = 0.0
roll             = 0.0
head_droop_log   = deque(maxlen=PERCLOS_WINDOW)

fatigue_index    = 0.0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Started.  터미널에서 입력: S=CSV저장  T=학습  L=피로도로그  Q=종료")


# ════════════════════════════════════════════════════════════════
#  메인 루프
# ════════════════════════════════════════════════════════════════
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame,1)   # 좌우 반전 (거울 모드)
    fh, fw  = frame.shape[:2]
    elapsed = time.time()-start_time
    overlay = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mpi)

    # 키 입력 처리
    while _key_queue:
        ch = _key_queue.popleft()
        if ch in ('q', '\x1b'):
            running = False; break
        elif ch == 's':
            if collected_rows: save_csv(collected_rows)
            else: print("No data yet.")
        elif ch == 't':
            if os.path.exists(CSV_PATH):
                threading.Thread(target=train_thread_fn, daemon=True).start()
            else: print("No CSV. Press S first.")
        elif ch == 'l':
            save_fatigue_log()
    if not running:
        break

    if res.face_landmarks:
        lms = res.face_landmarks[0]

        # EAR 계산 (좌우 평균 + 고개 방향 보정)
        le  = get_3d_ear(lms, LEFT_EYE,  fw, fh)
        re  = get_3d_ear(lms, RIGHT_EYE, fw, fh)
        cor = get_head_pose_correction(lms)
        ear = ((le+re)/2.0)*cor
        ear_window.append(ear); ear_seq.append(ear)

        mar = get_mar(lms, fw, fh)
        pitch, yaw_angle, roll = get_head_pose(lms, fw, fh)
        is_drooping = (pitch > PITCH_THRESH) or (abs(yaw_angle) > YAW_THRESH)
        head_droop_log.append(1 if is_drooping else 0)

        # 캘리브레이션: 처음 60프레임 동안 개인 EAR 기준값 측정
        if not calibrated:
            calib_ears.append(ear)
            prog = len(calib_ears)/CALIBRATION_FRAMES
            bw   = int(480*prog)
            cx,cy= fw//2, fh//2
            cv2.rectangle(overlay,(cx-240,cy-40),(cx+240,cy+40),(30,30,30),-1)
            cv2.rectangle(overlay,(cx-220,cy-14),(cx+220,cy+14),(55,55,55),-1)
            if bw>0:
                cv2.rectangle(overlay,(cx-220,cy-14),(cx-220+bw,cy+14),(0,180,90),-1)
            cv2.putText(overlay,f"Calibrating... {int(prog*100)}%",
                        (cx-130,cy-22),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),2,cv2.LINE_AA)
            if len(calib_ears)>=CALIBRATION_FRAMES:
                ear_open_baseline  = np.percentile(calib_ears,80)   # 상위 80% = 눈 뜬 기준
                personal_threshold = ear_open_baseline*0.75          # 기준의 75% 미만 → 눈 감음
                calibrated         = True
                print(f"Calibration done  threshold={personal_threshold:.3f}")
            cv2.imshow("Fatigue Detector", overlay)
            cv2.waitKey(1)
            continue

        # 드리프트 보정: 시간이 지나면서 EAR 기준값이 조금씩 바뀌는 것을 실시간 보정
        if len(ear_window)>=30:
            ro = np.percentile(list(ear_window),80)
            if ro > ear_open_baseline*0.5:
                personal_threshold = ro*0.75

        slope     = (ear-prev_ear) if prev_ear is not None else 0.0
        is_closed = ear < personal_threshold

        # 상태 머신: Normal → Onset → Valley → Offset 순서 강제
        # 기존 방식은 매 프레임 독립 판정이라 Onset 중 잠깐 기울기 0이 되면 Valley로 오분류됨
        # 상태를 기억해서 한 방향으로만 전이되도록 수정
        if not is_closed:
            blink_stage  = "Normal"
            _blink_phase = "none"
        else:
            if _blink_phase == "none":
                # 눈이 막 감겼을 때: 기울기가 음수면 Onset, 아니면 바로 Valley
                if slope < -SLOPE_THRESH:
                    blink_stage  = "Onset"
                    _blink_phase = "onset"
                else:
                    blink_stage  = "Valley"
                    _blink_phase = "valley"
            elif _blink_phase == "onset":
                if slope < -SLOPE_THRESH:
                    blink_stage = "Onset"           # 계속 감기는 중
                else:
                    blink_stage  = "Valley"         # 기울기가 0 근처 → 정점 도달
                    _blink_phase = "valley"
            elif _blink_phase == "valley":
                if slope > SLOPE_THRESH:
                    blink_stage  = "Offset"         # 뜨기 시작
                    _blink_phase = "offset"
                else:
                    blink_stage = "Valley"          # 아직 정점 구간
            elif _blink_phase == "offset":
                blink_stage = "Offset"              # 뜨는 중 → 눈 열릴 때까지 Offset 유지

        stage_history.append(blink_stage)
        stage_frame_count[blink_stage] += 1

        # 피처 생성 및 데이터 수집
        feat     = make_feature(ear_seq, personal_threshold, ear_open_baseline)
        rule_lbl = STAGE_MAP[blink_stage]
        collected_rows.append((feat, rule_lbl))

        # AI 추론 (모델이 로드된 경우)
        with _model_lock:
            ready = model_ready
        if ready:
            with _model_lock:
                ai_lbl = model.predict(feat)
            ai_stage = STAGE_LABELS[ai_lbl]

        # 눈 감는/뜨는 속도 기록 (피로도 계산에 활용)
        if slope < -SLOPE_THRESH: closing_slopes.append(abs(slope))
        elif slope > SLOPE_THRESH: opening_slopes.append(abs(slope))

        # 깜빡임 감지 및 완전성 판별
        if is_closed:
            blink_consec += 1
            blink_ear_min = min(blink_ear_min, ear)
            in_blink      = True
            eye_closed_log.append(1)
        else:
            if in_blink and blink_consec >= BLINK_CONSEC_FRAMES:
                blink_total += 1
                c = compute_completeness(ear_open_baseline, blink_ear_min)
                lbl,_ = classify_completeness(c)
                if lbl=="Incomplete": incomplete_total+=1
                elif lbl=="Micro":    micro_total+=1
            in_blink=False; blink_ear_min=1.0; blink_consec=0
            eye_closed_log.append(0)
        prev_ear = ear

        # 하품 감지: MAR이 임계값을 일정 프레임 이상 초과하면 하품으로 카운트
        if mar > MAR_YAWN_THRESH:
            yawn_consec+=1; in_yawn=True
        else:
            if in_yawn and yawn_consec>=MAR_CONSEC_FRAMES: yawn_total+=1
            in_yawn=False; yawn_consec=0

        # 각종 지표 계산
        perclos          = sum(eye_closed_log)/max(len(eye_closed_log),1)
        avg_c            = np.mean(closing_slopes) if closing_slopes else 0.01
        avg_o            = np.mean(opening_slopes) if opening_slopes else 0.01
        fatigue_ratio    = avg_o/(avg_c+1e-6)   # 1보다 작으면 뜨는 속도 < 감는 속도 → 피로
        bpm              = blink_total/max(elapsed/60.0,1/60.0)
        inc_ratio        = incomplete_total/max(blink_total,1)
        yawn_rpm         = yawn_total/max(elapsed/60.0,1/60.0)
        head_droop_ratio = sum(head_droop_log)/max(len(head_droop_log),1)

        fatigue_index    = compute_fatigue_index(perclos,inc_ratio,yawn_rpm,head_droop_ratio)
        fi_label, fi_col = fatigue_level(fatigue_index)

        # 5분마다 피로도 스냅샷 저장
        if time.time()-last_log_time >= LOG_INTERVAL_SEC:
            fatigue_log.append([round(elapsed,1),round(fatigue_index,4),
                                 round(perclos,4),round(inc_ratio,4),
                                 round(yawn_rpm,4),round(head_droop_ratio,4)])
            last_log_time=time.time()
            print(f"[LOG] t={elapsed:.0f}s FI={fatigue_index:.3f} "
                  f"PERCLOS={perclos:.3f} INC={inc_ratio:.3f} "
                  f"YAWN={yawn_rpm:.2f}/min HEAD={head_droop_ratio:.3f}")

        # 눈·입 랜드마크 시각화
        for i in LEFT_EYE+RIGHT_EYE:
            lm=lms[i]; cv2.circle(overlay,(int(lm.x*fw),int(lm.y*fh)),2,(0,230,180),-1)
        for i in MOUTH_IDX:
            lm=lms[i]; cv2.circle(overlay,(int(lm.x*fw),int(lm.y*fh)),2,(255,180,0),-1)

        # 코 끝에 고개 방향 화살표 그리기
        nl=lms[NOSE_TIP]; nx,ny=int(nl.x*fw),int(nl.y*fh)
        pr,yr=math.radians(pitch),math.radians(yaw_angle)
        ax=int(nx+60*math.sin(yr)); ay=int(ny-60*math.cos(yr)*math.cos(pr))
        cv2.arrowedLine(overlay,(nx,ny),(ax,ay),
                        (60,60,220) if is_drooping else (0,200,80),2,tipLength=0.3)

        # ── 왼쪽 정보 패널 그리기 ────────────────────────────────
        panel = np.full((fh,PANEL_W,3),(18,18,18),dtype=np.uint8)
        py    = 0

        def section(title, gap=8):
            global py
            py+=gap; divider(panel,py); py+=2
            txt(panel,title,14,py+15,(110,110,110),0.40); py+=20

        py=12
        txt(panel,"FATIGUE MONITOR",14,py+18,(90,200,255),0.58,2); py+=20
        txt(panel,f"{int(elapsed//60):02d}:{int(elapsed%60):02d}",
            14,py+14,(90,90,90),0.40); py+=16

        section("FATIGUE INDEX",gap=6)
        cv2.rectangle(panel,(14,py),(PANEL_W-14,py+38),(28,28,28),-1)
        cv2.rectangle(panel,(14,py),(18,py+38),fi_col,-1)
        txt(panel,fi_label,28,py+16,fi_col,0.62,2)
        txt(panel,f"{fatigue_index*100:.1f}%",28,py+34,fi_col,0.48); py+=44
        draw_bar(panel,14,py,PANEL_W-28,8,fatigue_index,1.0,fi_col); py+=14

        section("EAR")
        ear_col=(0,200,80) if not is_closed else (60,60,220)
        txt(panel,f"Current {ear:.3f}  Thresh {personal_threshold:.3f}",
            14,py+16,(210,210,210),0.42); py+=20
        draw_bar(panel,14,py,PANEL_W-28,8,ear,ear_open_baseline,ear_col); py+=14

        section("BLINK STATS")
        txt(panel,f"Total {blink_total}  BPM {bpm:.1f}",14,py+16); py+=20
        ic=(0,165,255) if inc_ratio>0.3 else (160,160,160)
        txt(panel,f"Incomplete {incomplete_total} ({inc_ratio*100:.0f}%)  Micro {micro_total}",
            14,py+16,ic,0.42); py+=20

        section("PERCLOS")
        pc=(60,60,220) if perclos>PERCLOS_THRESHOLD else (0,200,80)
        txt(panel,f"{perclos*100:.1f}%  (alert>{PERCLOS_THRESHOLD*100:.0f}%)",
            14,py+16,pc,0.44); py+=20
        draw_bar(panel,14,py,PANEL_W-28,8,perclos,0.30,pc)
        tx=14+int(PERCLOS_THRESHOLD/0.30*(PANEL_W-28))
        cv2.line(panel,(tx,py-3),(tx,py+11),(180,60,60),2); py+=14

        section("MAR / YAWN")
        mc=(60,60,220) if mar>MAR_YAWN_THRESH else (0,200,80)
        txt(panel,f"MAR {mar:.3f}  Yawns {yawn_total} ({yawn_rpm:.1f}/min)",
            14,py+16,mc,0.42); py+=20
        draw_bar(panel,14,py,PANEL_W-28,8,mar,1.0,mc); py+=14
        if in_yawn:
            txt(panel,"YAWNING",14,py+14,(60,60,220),0.50,2); py+=18

        section("HEAD POSE")
        hc=(60,60,220) if is_drooping else (0,200,80)
        txt(panel,f"Pitch {pitch:+.1f}  Yaw {yaw_angle:+.1f}  Roll {roll:+.1f}",
            14,py+16,hc,0.42); py+=20
        txt(panel,f"Droop {head_droop_ratio*100:.1f}%",
            14,py+14,(150,150,150),0.38); py+=18
        draw_bar(panel,14,py,PANEL_W-28,8,head_droop_ratio,0.5,hc); py+=14

        section("EYE FATIGUE")
        fc=(60,60,220) if fatigue_ratio<FATIGUE_RATIO_THRESH else (0,200,80)
        txt(panel,f"Open/Close {fatigue_ratio:.2f}  (alert<{FATIGUE_RATIO_THRESH})",
            14,py+16,fc,0.42); py+=20

        section("EYE STATUS")
        if is_closed:
            cv2.rectangle(panel,(14,py),(PANEL_W-14,py+26),(35,15,15),-1)
            cv2.rectangle(panel,(14,py),(18,py+26),(60,60,220),-1)
            txt(panel,"EYE CLOSED",28,py+20,(80,80,230),0.56,2)
        else:
            cv2.rectangle(panel,(14,py),(PANEL_W-14,py+26),(15,35,15),-1)
            cv2.rectangle(panel,(14,py),(18,py+26),(0,200,80),-1)
            txt(panel,"EYE OPEN",28,py+20,(0,200,80),0.56,2)
        py+=32

        if py+50 < fh-30:
            section("STAGE [Rule]",gap=6)
            sc=STAGE_COLORS[blink_stage]
            cv2.rectangle(panel,(14,py),(PANEL_W-14,py+24),(30,30,30),-1)
            cv2.rectangle(panel,(14,py),(18,py+24),sc,-1)
            txt(panel,blink_stage.upper(),28,py+18,sc,0.56,2); py+=30

        if py+50 < fh-30:
            section("STAGE [AI]",gap=6)
            if model_ready:
                asc=STAGE_COLORS[ai_stage]
                cv2.rectangle(panel,(14,py),(PANEL_W-14,py+24),(30,30,30),-1)
                cv2.rectangle(panel,(14,py),(18,py+24),asc,-1)
                txt(panel,ai_stage.upper(),28,py+18,asc,0.56,2); py+=30
            else:
                txt(panel,training_msg or "T=train",14,py+16,(100,100,100),0.40); py+=20

        txt(panel,f"Rows:{len(collected_rows)}  S=csv T=train L=log Q=quit",
            14,fh-14,(70,70,70),0.36)
        overlay[:,:PANEL_W]=panel

        # 경고 배너 (화면 하단에 표시)
        warnings=[]
        if fi_label=="DANGER":   warnings.append(("HIGH FATIGUE DETECTED",(60,40,220)))
        elif fi_label=="CAUTION":warnings.append(("Fatigue increasing",   (60,100,210)))
        if perclos>PERCLOS_THRESHOLD: warnings.append(("PERCLOS alert!",(60,60,200)))
        if yawn_rpm>2:           warnings.append(("Yawning frequently!",  (60,80,210)))
        if is_drooping:          warnings.append(("Head drooping",        (60,80,200)))
        if inc_ratio>0.5 and blink_total>5:
            warnings.append(("Too many incomplete blinks",(60,100,210)))
        for i,(msg,col) in enumerate(warnings):
            yw=fh-16-i*32
            if yw<fh//2: break
            cv2.rectangle(overlay,(PANEL_W+10,yw-20),(fw-10,yw+8),(28,28,28),-1)
            cv2.rectangle(overlay,(PANEL_W+10,yw-20),(PANEL_W+15,yw+8),col,-1)
            cv2.putText(overlay,msg,(PANEL_W+22,yw),
                        cv2.FONT_HERSHEY_SIMPLEX,0.50,(210,210,210),1,cv2.LINE_AA)

    else:
        # 얼굴이 감지되지 않을 때
        overlay[:,:PANEL_W]=(18,18,18)
        cv2.putText(overlay,"No face detected",
                    (PANEL_W+(fw-PANEL_W)//2-140,fh//2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(80,80,180),2,cv2.LINE_AA)

    cv2.imshow("Fatigue Detector", overlay)
    key = cv2.waitKey(1) & 0xFF
    if not _is_tty and key != 255:   # PyCharm 환경: OpenCV 창에서도 키 입력 받기
        ch = chr(key).lower() if key < 128 else ''
        if ch:
            _key_queue.append(ch)
    if cv2.getWindowProperty("Fatigue Detector", cv2.WND_PROP_VISIBLE) < 1:
        break   # X 버튼으로 창 닫으면 종료

# 종료 처리
_stop_input = True
cap.release()
cv2.destroyAllWindows()
detector.close()
_input_thread.join(timeout=0.3)  # 입력 스레드가 select 타임아웃 후 자연 종료되길 기다림
print("Done.")
