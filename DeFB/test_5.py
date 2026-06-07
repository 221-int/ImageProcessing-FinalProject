"""
Fatigue Detector v5 - Avatar Mode
─────────────────────────────────────────────────────────────────
v4와 동일한 분석 로직 + 카메라 화면 대신 캐릭터 아바타 표시
  - EAR → 캐릭터 눈 크기
  - MAR → 캐릭터 입 크기
  - Roll → 캐릭터 고개 기울기
  - 피로도 → 캐릭터 표정/색상 변화

조작키 (PyCharm 콘솔에서 입력 후 엔터)
  S : 깜빡임 데이터 CSV 저장
  T : DNN 학습 시작
  L : 피로도 로그 CSV 저장
  Q : 종료
"""

import cv2
import numpy as np
from collections import deque
import time, os, csv, threading, math, sys, tty, termios, select

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ════════════════════════════════════════════════════════════════
#  상수 (test_4와 동일)
# ════════════════════════════════════════════════════════════════
LEFT_EYE    = [362, 385, 387, 263, 373, 380]
RIGHT_EYE   = [33,  160, 158, 133, 153, 144]
MOUTH_IDX   = [61, 291, 13, 14, 78, 308]
NOSE_TIP    = 1;  CHIN = 152
LEFT_EYE_L  = 33; RIGHT_EYE_R = 263
LEFT_MOUTH  = 61; RIGHT_MOUTH = 291

CALIBRATION_FRAMES   = 60
SLIDING_WINDOW       = 300
BLINK_CONSEC_FRAMES  = 2
PERCLOS_WINDOW       = 900
PERCLOS_THRESHOLD    = 0.15
FATIGUE_RATIO_THRESH = 0.6
PANEL_W              = 340
SLOPE_THRESH         = 0.005
MAR_YAWN_THRESH      = 0.6
MAR_CONSEC_FRAMES    = 15
PITCH_THRESH         = 15.0
YAW_THRESH           = 25.0
LOG_INTERVAL_SEC     = 300
W_PERCLOS = 0.35; W_INC_BLINK = 0.25; W_YAWN = 0.20; W_HEAD = 0.20

SEQUENCE_LEN = 10
STAGE_LABELS = ["Normal", "Onset", "Valley", "Offset"]
STAGE_MAP    = {s: i for i, s in enumerate(STAGE_LABELS)}
STAGE_COLORS = {
    "Normal": (0,200,80), "Onset": (0,165,255),
    "Valley": (60,60,220), "Offset": (180,120,0),
}
FEATURE_LEN = SEQUENCE_LEN + 4
CSV_PATH    = "blink_data.csv"
MODEL_PATH  = "blink_model.npz"
FATIGUE_LOG = "fatigue_log.csv"

# ════════════════════════════════════════════════════════════════
#  키 입력 스레드
# ════════════════════════════════════════════════════════════════
_key_queue  = deque(maxlen=10)
_stop_input = False
_is_tty     = sys.stdin.isatty()

def _key_listener():
    if _is_tty:
        fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not _stop_input:
                if select.select([sys.stdin],[],[],0.05)[0]:
                    _key_queue.append(sys.stdin.read(1).lower())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    else:
        print("PyCharm 모드: 명령 입력 후 엔터 (s=저장 t=학습 l=로그 q=종료)")
        while not _stop_input:
            try:
                if select.select([sys.stdin],[],[],0.1)[0]:
                    for ch in sys.stdin.readline().strip().lower():
                        _key_queue.append(ch)
            except (EOFError, OSError): break

_input_thread = threading.Thread(target=_key_listener, daemon=True)
_input_thread.start()

# ── MediaPipe ────────────────────────────────────────────────────
base_options = python.BaseOptions(
    model_asset_path='/Users/hanool/PycharmProjects/PythonProject/DeFB/face_landmarker.task'
)
detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
)

# ════════════════════════════════════════════════════════════════
#  분석 함수 (test_4와 동일)
# ════════════════════════════════════════════════════════════════
def get_mar(landmarks, img_w, img_h):
    def pt(idx): lm=landmarks[idx]; return np.array([lm.x*img_w, lm.y*img_h])
    left,right = pt(MOUTH_IDX[0]),pt(MOUTH_IDX[1])
    top1,bot1  = pt(MOUTH_IDX[2]),pt(MOUTH_IDX[3])
    top2,bot2  = pt(MOUTH_IDX[4]),pt(MOUTH_IDX[5])
    return (np.linalg.norm(top1-bot1)+np.linalg.norm(top2-bot2))/(2*np.linalg.norm(left-right)+1e-6)

FACE_3D = np.array([[0,0,0],[0,-63.6,-12.5],[-43.3,32.7,-26],[43.3,32.7,-26],
                    [-28.9,-28.9,-24.1],[28.9,-28.9,-24.1]], dtype=np.float64)
POSE_LM_IDX = [NOSE_TIP,CHIN,LEFT_EYE_L,RIGHT_EYE_R,LEFT_MOUTH,RIGHT_MOUTH]

def get_head_pose(landmarks, img_w, img_h):
    face_2d = np.array([[landmarks[i].x*img_w,landmarks[i].y*img_h] for i in POSE_LM_IDX],dtype=np.float64)
    focal   = img_w
    cam_mat = np.array([[focal,0,img_w/2],[0,focal,img_h/2],[0,0,1]],dtype=np.float64)
    ok,rvec,_ = cv2.solvePnP(FACE_3D,face_2d,cam_mat,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.,0.,0.
    rmat,_ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0,0]**2+rmat[1,0]**2)
    if sy>1e-6:
        return (math.degrees(math.atan2(rmat[2,1],rmat[2,2])),
                math.degrees(math.atan2(-rmat[2,0],sy)),
                math.degrees(math.atan2(rmat[1,0],rmat[0,0])))
    return math.degrees(math.atan2(-rmat[1,2],rmat[1,1])),math.degrees(math.atan2(-rmat[2,0],sy)),0.

def compute_fatigue_index(perclos,inc_ratio,yawn_rpm,head_droop_ratio):
    return min(W_PERCLOS*min(perclos/PERCLOS_THRESHOLD,1)+W_INC_BLINK*inc_ratio
               +W_YAWN*min(yawn_rpm/3,1)+W_HEAD*head_droop_ratio, 1.0)

def fatigue_level(fi):
    if fi<0.3:   return "NORMAL",  (0,200,80)
    elif fi<0.6: return "CAUTION", (0,165,255)
    else:        return "DANGER",  (60,60,220)

def get_3d_ear(landmarks,eye_indices,img_w,img_h):
    pts=[np.array([landmarks[i].x*img_w,landmarks[i].y*img_h,landmarks[i].z*img_w]) for i in eye_indices]
    return (np.linalg.norm(pts[1]-pts[5])+np.linalg.norm(pts[2]-pts[4]))/(2*np.linalg.norm(pts[0]-pts[3])+1e-6)

def get_head_pose_correction(landmarks):
    return 1.0+0.08*(1.0-min(abs(landmarks[130].x-landmarks[359].x)/0.4,1.0))

def compute_completeness(ear_open,ear_min):
    return 0. if ear_open<1e-6 else (ear_open-ear_min)/ear_open

def classify_completeness(c):
    if c>0.9: return "Complete",(0,200,0)
    elif c>0.5: return "Incomplete",(0,165,255)
    else: return "Micro",(0,0,255)

def draw_bar(img,x,y,w,h,value,max_val,cf,cb=(50,50,50)):
    cv2.rectangle(img,(x,y),(x+w,y+h),cb,-1)
    f=int(min(value/max(max_val,1e-6),1)*w)
    if f>0: cv2.rectangle(img,(x,y),(x+f,y+h),cf,-1)

def txt(img,text,x,y,color=(210,210,210),scale=0.50,bold=1):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,bold,cv2.LINE_AA)

def divider(img,y):
    cv2.line(img,(14,y),(PANEL_W-14,y),(50,50,50),1)

def make_feature(ear_seq,threshold,baseline):
    seq=list(ear_seq)
    if len(seq)<SEQUENCE_LEN: seq=[seq[0]]*(SEQUENCE_LEN-len(seq))+seq
    norm=[e/max(baseline,1e-6) for e in seq]
    s1=seq[-1]-seq[-2]; s2=(seq[-1]-seq[-3])/2; s3=(seq[-1]-seq[-4])/3
    return np.array(norm+[s1/max(baseline,1e-6),s2/max(baseline,1e-6),
                           s3/max(baseline,1e-6),1. if seq[-1]<threshold else 0.],dtype=np.float32)

# ════════════════════════════════════════════════════════════════
#  TinyDNN (test_4와 동일)
# ════════════════════════════════════════════════════════════════
class TinyDNN:
    def __init__(self,in_dim=FEATURE_LEN,h1=64,h2=32,out=4):
        rng=np.random.default_rng(42)
        self.W1=rng.standard_normal((in_dim,h1)).astype(np.float32)*np.sqrt(2/in_dim)
        self.b1=np.zeros(h1,dtype=np.float32)
        self.W2=rng.standard_normal((h1,h2)).astype(np.float32)*np.sqrt(2/h1)
        self.b2=np.zeros(h2,dtype=np.float32)
        self.W3=rng.standard_normal((h2,out)).astype(np.float32)*np.sqrt(2/h2)
        self.b3=np.zeros(out,dtype=np.float32)
    def _relu(self,x): return np.maximum(0,x)
    def _softmax(self,x):
        e=np.exp(x-x.max(axis=-1,keepdims=True)); return e/e.sum(axis=-1,keepdims=True)
    def forward(self,X):
        self.a0=X; self.z1=X@self.W1+self.b1; self.a1=self._relu(self.z1)
        self.z2=self.a1@self.W2+self.b2; self.a2=self._relu(self.z2)
        self.z3=self.a2@self.W3+self.b3; self.a3=self._softmax(self.z3); return self.a3
    def predict(self,x): return int(np.argmax(self.forward(x[np.newaxis])[0]))
    def _ce(self,p,y): return -np.mean(np.sum(y*np.log(p+1e-9),axis=1))
    def train(self,X,y,epochs=300,lr=0.005,batch=128):
        nc=self.W3.shape[1]; n=len(X); sp=int(n*0.8); pm=np.random.permutation(n)
        Xtr,ytr=X[pm[:sp]],y[pm[:sp]]; Xval,yval=X[pm[sp:]],y[pm[sp:]]
        yoh=np.eye(nc,dtype=np.float32)[ytr]; bv,bw=0.,self._gw()
        for ep in range(epochs):
            idx=np.random.permutation(len(Xtr)); el=0.
            for s in range(0,len(Xtr),batch):
                bx=Xtr[idx[s:s+batch]]; by=yoh[idx[s:s+batch]]
                p=self.forward(bx); el+=self._ce(p,by)
                dz3=(p-by)/len(bx)
                dW3=self.a2.T@dz3; db3=dz3.sum(0)
                dz2=(dz3@self.W3.T)*(self.z2>0); dW2=self.a1.T@dz2; db2=dz2.sum(0)
                dz1=(dz2@self.W2.T)*(self.z1>0); dW1=self.a0.T@dz1; db1=dz1.sum(0)
                self.W3-=lr*dW3; self.b3-=lr*db3
                self.W2-=lr*dW2; self.b2-=lr*db2
                self.W1-=lr*dW1; self.b1-=lr*db1
            if (ep+1)%50==0:
                ta=np.mean(np.argmax(self.forward(Xtr),axis=1)==ytr)
                va=np.mean(np.argmax(self.forward(Xval),axis=1)==yval)
                print(f"  ep{ep+1}/{epochs} loss={el:.3f} tr={ta*100:.1f}% val={va*100:.1f}%")
                if va>bv: bv=va; bw=self._gw()
        self._sw(bw); print(f"Best val={bv*100:.1f}%")
    def _gw(self): return tuple(w.copy() for w in [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3])
    def _sw(self,w): self.W1,self.b1,self.W2,self.b2,self.W3,self.b3=[x.copy() for x in w]
    def save(self,path):
        np.savez(path,W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,W3=self.W3,b3=self.b3)
        print(f"Model saved → {path}")
    def load(self,path):
        d=np.load(path)
        self.W1,self.b1=d["W1"],d["b1"]; self.W2,self.b2=d["W2"],d["b2"]; self.W3,self.b3=d["W3"],d["b3"]
        print(f"Model loaded ← {path}")

# ── 데이터/학습 유틸 ─────────────────────────────────────────────
collected_rows=[]; fatigue_log=[]

def save_csv(rows,path=CSV_PATH):
    cols=[f"f{i}" for i in range(FEATURE_LEN)]
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(cols+["label"])
        for feat,lbl in rows: w.writerow(list(feat)+[lbl])
    print(f"CSV saved ({len(rows)} rows) → {path}")

def load_csv(path=CSV_PATH):
    X,y=[],[]; cols=[f"f{i}" for i in range(FEATURE_LEN)]
    with open(path,newline="") as f:
        for row in csv.DictReader(f):
            X.append([float(row[k]) for k in cols]); y.append(int(row["label"]))
    return np.array(X,np.float32),np.array(y,np.int32)

def balance(X,y):
    c=np.bincount(y,minlength=4); nm=max(c[1],c[2],c[3])
    if nm<5:
        print(f"  [경고] 깜빡임 샘플 부족. 원본으로 학습합니다.")
        p=np.random.permutation(len(X)); return X[p],y[p]
    nc=min(c[0],nm*3); tg=max(nm,100); Xo,yo=[],[]
    for cls in range(4):
        idx=np.where(y==cls)[0]
        if not len(idx): continue
        cap=nc if cls==0 else tg
        ch=(np.random.choice(idx,cap,replace=False) if len(idx)>=cap
            else np.concatenate([idx,np.random.choice(idx,cap-len(idx),replace=True)]))
        Xo.append(X[ch]); yo.append(y[ch])
    Xb=np.concatenate(Xo); yb=np.concatenate(yo); p=np.random.permutation(len(Xb))
    return Xb[p],yb[p]

def save_fatigue_log():
    with open(FATIGUE_LOG,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["time_sec","fatigue_index","perclos","inc_ratio","yawn_rpm","head_droop"])
        for row in fatigue_log: w.writerow(row)
    print(f"Fatigue log saved ({len(fatigue_log)} entries) → {FATIGUE_LOG}")

_model_lock=threading.Lock(); model=TinyDNN(); model_ready=False; training_msg=""

def train_thread_fn():
    global model,model_ready,training_msg
    training_msg="Training..."
    try:
        X,y=load_csv(); Xb,yb=balance(X,y)
        print(f"Balanced {len(Xb)} rows  cls={np.bincount(yb,minlength=4)}")
        nm=TinyDNN(); nm.train(Xb,yb); nm.save(MODEL_PATH)
        with _model_lock: model=nm; model_ready=True; training_msg=f"AI ready! ({len(X)} samples)"
    except Exception as e: training_msg=f"Train error: {e}"; print(training_msg)

if os.path.exists(MODEL_PATH):
    try:
        d=np.load(MODEL_PATH)
        if d["W1"].shape[0]==FEATURE_LEN:
            model.load(MODEL_PATH); model_ready=True; training_msg="Model pre-loaded"
        else: print(f"Model dim mismatch. Skipping.")
    except Exception as e: print(f"Model load failed: {e}")


# ════════════════════════════════════════════════════════════════
#  캐릭터 아바타 그리기
#  EAR → 눈 크기, MAR → 입 크기, roll → 고개 기울기, fi → 표정
# ════════════════════════════════════════════════════════════════
# ── Claude 픽셀 캐릭터 3프레임 로드 + 미리 리사이즈 ─────────────
# 눈감음(0) / 눈중간(1) / 눈뜸(2) 순서
_BASE = '/Users/hanool/PycharmProjects/PythonProject/DeFB'
_frame_files = ['눈감음.png', '눈중간.png', '눈뜸.png']
_AVATAR_SIZE = 520   # Claude 창 크기 (정사각형)

_frames_raw = []
for fname in _frame_files:
    img = cv2.imread(f'{_BASE}/{fname}')
    if img is None:
        raise FileNotFoundError(f"{fname} 없음 — DeFB 폴더에 있는지 확인하세요")
    _frames_raw.append(img)

# 스프라이트를 창 크기에 맞게 한 번만 리사이즈 (매 프레임 리사이즈 방지)
def _presize(img, canvas_size=_AVATAR_SIZE, pad=20):
    h, w = img.shape[:2]
    scale = min((canvas_size - pad) / w, (canvas_size - pad) / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ox = (canvas_size - nw) // 2
    oy = (canvas_size - nh) // 2
    return resized, ox, oy   # (이미지, x오프셋, y오프셋)

_frames = [_presize(f) for f in _frames_raw]   # [(img, ox, oy), ...]

# 아바타 캔버스 재사용 (매 프레임 np.zeros 방지)
_avatar_canvas = np.zeros((_AVATAR_SIZE, _AVATAR_SIZE, 3), dtype=np.uint8)


def draw_avatar(canvas, cx, cy, ear_ratio, mar_val, roll_deg, fi_label, fi_col, is_drooping):
    """
    3-프레임 스프라이트 전환으로 눈 깜빡임 표현
      frame0: 거의 감김 (ear_ratio < 0.35)
      frame1: 반쯤 뜸  (ear_ratio < 0.70)
      frame2: 완전히 뜸 (ear_ratio >= 0.70)
    """
    h, w = canvas.shape[:2]

    # 배경
    bg = {"NORMAL": (25,32,25), "CAUTION": (25,30,42), "DANGER": (38,22,22)}
    canvas[:] = bg.get(fi_label, (22,22,28))

    # EAR 비율로 프레임 선택 (미리 리사이즈된 프레임 사용)
    if ear_ratio < 0.35:
        sprite, ox, oy = _frames[0]
    elif ear_ratio < 0.70:
        sprite, ox, oy = _frames[1]
    else:
        sprite, ox, oy = _frames[2]

    new_h, new_w = sprite.shape[:2]
    canvas[oy:oy+new_h, ox:ox+new_w] = sprite

    # 피로도 텍스트
    cv2.putText(canvas, fi_label, (w//2 - 50, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fi_col, 2, cv2.LINE_AA)

    # 땀방울 (고개 숙임)
    if is_drooping:
        sx, sy = ox + new_w - 5, oy + 30
        cv2.ellipse(canvas, (sx, sy),    (8, 14), 0, 0, 360, (200,160,60), -1)
        cv2.ellipse(canvas, (sx, sy-18), (5,  8), 0, 0, 360, (200,160,60), -1)


# ════════════════════════════════════════════════════════════════
#  메인 상태 변수
# ════════════════════════════════════════════════════════════════
calib_ears=[]; personal_threshold=None; ear_open_baseline=None; calibrated=False
ear_window=deque(maxlen=SLIDING_WINDOW); ear_seq=deque(maxlen=SEQUENCE_LEN)
eye_closed_log=deque(maxlen=PERCLOS_WINDOW)
closing_slopes=deque(maxlen=20); opening_slopes=deque(maxlen=20)

blink_total=0; incomplete_total=0; micro_total=0
blink_consec=0; in_blink=False; blink_ear_min=1.0; prev_ear=None
start_time=time.time(); last_log_time=time.time()

blink_stage="Normal"; ai_stage="Normal"; _blink_phase="none"
stage_history=deque(maxlen=SLIDING_WINDOW)
stage_frame_count={s:0 for s in STAGE_LABELS}

mar=0.; yawn_consec=0; yawn_total=0; in_yawn=False
pitch=0.; yaw_angle=0.; roll=0.
head_droop_log=deque(maxlen=PERCLOS_WINDOW)
fatigue_index=0.
ear_ratio_smooth = 1.0   # EAR 비율 스무딩 (아바타 눈 떨림 방지)
mar_smooth       = 0.0   # MAR 스무딩
_frame_count     = 0     # head pose 스킵용 카운터

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Started.  S=저장  T=학습  L=로그  Q=종료")


# ════════════════════════════════════════════════════════════════
#  메인 루프
# ════════════════════════════════════════════════════════════════
running = True
while running:
    ret, frame = cap.read()
    if not ret: break

    frame  = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    elapsed = time.time() - start_time

    # 카메라 화면 그대로 사용
    overlay = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mpi)

    # 키 입력 처리
    while _key_queue:
        ch = _key_queue.popleft()
        if ch in ('q','\x1b'): running=False; break
        elif ch=='s':
            if collected_rows: save_csv(collected_rows)
            else: print("No data yet.")
        elif ch=='t':
            if os.path.exists(CSV_PATH): threading.Thread(target=train_thread_fn,daemon=True).start()
            else: print("No CSV. Press S first.")
        elif ch=='l': save_fatigue_log()
    if not running: break

    if res.face_landmarks:
        lms = res.face_landmarks[0]

        le  = get_3d_ear(lms,LEFT_EYE,fw,fh)
        re  = get_3d_ear(lms,RIGHT_EYE,fw,fh)
        cor = get_head_pose_correction(lms)
        ear = ((le+re)/2.0)*cor
        ear_window.append(ear); ear_seq.append(ear)

        mar = get_mar(lms,fw,fh)
        _frame_count += 1
        if _frame_count % 3 == 0:  # solvePnP 비용이 커서 3프레임마다 계산
            pitch,yaw_angle,roll = get_head_pose(lms,fw,fh)
        is_drooping = (pitch>PITCH_THRESH) or (abs(yaw_angle)>YAW_THRESH)
        head_droop_log.append(1 if is_drooping else 0)

        # 캘리브레이션
        if not calibrated:
            calib_ears.append(ear)
            prog = len(calib_ears)/CALIBRATION_FRAMES
            cx_c,cy_c = fw//2, fh//2
            cv2.rectangle(overlay,(cx_c-240,cy_c-40),(cx_c+240,cy_c+40),(30,30,30),-1)
            cv2.rectangle(overlay,(cx_c-220,cy_c-14),(cx_c+220,cy_c+14),(55,55,55),-1)
            bw_c = int(480*prog)
            if bw_c>0: cv2.rectangle(overlay,(cx_c-220,cy_c-14),(cx_c-220+bw_c,cy_c+14),(0,180,90),-1)
            cv2.putText(overlay,f"Calibrating... {int(prog*100)}%",
                        (cx_c-130,cy_c-22),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),2,cv2.LINE_AA)
            if len(calib_ears)>=CALIBRATION_FRAMES:
                ear_open_baseline  = np.percentile(calib_ears,80)
                personal_threshold = ear_open_baseline*0.75
                calibrated=True
                print(f"Calibration done  threshold={personal_threshold:.3f}")
            cv2.imshow("Fatigue Detector v5", overlay)
            cv2.waitKey(1); continue

        if len(ear_window)>=30:
            ro=np.percentile(list(ear_window),80)
            if ro>ear_open_baseline*0.5: personal_threshold=ro*0.75

        slope     = (ear-prev_ear) if prev_ear is not None else 0.
        is_closed = ear < personal_threshold

        # 상태 머신
        if not is_closed:
            blink_stage="Normal"; _blink_phase="none"
        else:
            if _blink_phase=="none":
                blink_stage,_blink_phase = ("Onset","onset") if slope<-SLOPE_THRESH else ("Valley","valley")
            elif _blink_phase=="onset":
                if slope<-SLOPE_THRESH: blink_stage="Onset"
                else: blink_stage="Valley"; _blink_phase="valley"
            elif _blink_phase=="valley":
                if slope>SLOPE_THRESH: blink_stage="Offset"; _blink_phase="offset"
                else: blink_stage="Valley"
            elif _blink_phase=="offset":
                blink_stage="Offset"

        stage_history.append(blink_stage); stage_frame_count[blink_stage]+=1

        feat     = make_feature(ear_seq,personal_threshold,ear_open_baseline)
        rule_lbl = STAGE_MAP[blink_stage]
        collected_rows.append((feat,rule_lbl))

        with _model_lock: ready=model_ready
        if ready:
            with _model_lock: ai_lbl=model.predict(feat)
            ai_stage=STAGE_LABELS[ai_lbl]

        if slope<-SLOPE_THRESH: closing_slopes.append(abs(slope))
        elif slope>SLOPE_THRESH: opening_slopes.append(abs(slope))

        if is_closed:
            blink_consec+=1; blink_ear_min=min(blink_ear_min,ear); in_blink=True; eye_closed_log.append(1)
        else:
            if in_blink and blink_consec>=BLINK_CONSEC_FRAMES:
                blink_total+=1; c=compute_completeness(ear_open_baseline,blink_ear_min)
                lbl,_=classify_completeness(c)
                if lbl=="Incomplete": incomplete_total+=1
                elif lbl=="Micro": micro_total+=1
            in_blink=False; blink_ear_min=1.; blink_consec=0; eye_closed_log.append(0)
        prev_ear=ear

        if mar>MAR_YAWN_THRESH: yawn_consec+=1; in_yawn=True
        else:
            if in_yawn and yawn_consec>=MAR_CONSEC_FRAMES: yawn_total+=1
            in_yawn=False; yawn_consec=0

        perclos       = sum(eye_closed_log)/max(len(eye_closed_log),1)
        avg_c         = np.mean(closing_slopes) if closing_slopes else 0.01
        avg_o         = np.mean(opening_slopes) if opening_slopes else 0.01
        fatigue_ratio = avg_o/(avg_c+1e-6)
        bpm           = blink_total/max(elapsed/60.,1/60.)
        inc_ratio     = incomplete_total/max(blink_total,1)
        yawn_rpm      = yawn_total/max(elapsed/60.,1/60.)
        head_droop_ratio = sum(head_droop_log)/max(len(head_droop_log),1)

        fatigue_index    = compute_fatigue_index(perclos,inc_ratio,yawn_rpm,head_droop_ratio)
        fi_label, fi_col = fatigue_level(fatigue_index)

        if time.time()-last_log_time>=LOG_INTERVAL_SEC:
            fatigue_log.append([round(elapsed,1),round(fatigue_index,4),round(perclos,4),
                                 round(inc_ratio,4),round(yawn_rpm,4),round(head_droop_ratio,4)])
            last_log_time=time.time()

        # EAR/MAR 스무딩 (아바타가 덜 떨리도록)
        raw_ear_ratio    = ear / max(ear_open_baseline, 1e-6)
        ear_ratio_smooth = ear_ratio_smooth * 0.6 + raw_ear_ratio * 0.4
        mar_smooth       = mar_smooth * 0.5 + mar * 0.5

        # ── 눈·입 랜드마크 점 시각화 ─────────────────────────────
        for i in LEFT_EYE+RIGHT_EYE:
            lm=lms[i]; cv2.circle(overlay,(int(lm.x*fw),int(lm.y*fh)),2,(0,230,180),-1)
        for i in MOUTH_IDX:
            lm=lms[i]; cv2.circle(overlay,(int(lm.x*fw),int(lm.y*fh)),2,(255,180,0),-1)

        # 코 끝 고개 방향 화살표
        nl=lms[NOSE_TIP]; nx,ny=int(nl.x*fw),int(nl.y*fh)
        pr,yr=math.radians(pitch),math.radians(yaw_angle)
        ax=int(nx+60*math.sin(yr)); ay=int(ny-60*math.cos(yr)*math.cos(pr))
        cv2.arrowedLine(overlay,(nx,ny),(ax,ay),
                        (60,60,220) if is_drooping else (0,200,80),2,tipLength=0.3)

        # ── 아바타: 별도 창에 그리기 (재사용 캔버스) ──────────────
        draw_avatar(
            _avatar_canvas,
            cx       = 260,
            cy       = 260,
            ear_ratio= ear_ratio_smooth,
            mar_val  = mar_smooth,
            roll_deg = roll,
            fi_label = fi_label,
            fi_col   = fi_col,
            is_drooping = is_drooping,
        )
        cv2.imshow("Claude", _avatar_canvas)

        # ── 정보 패널 (왼쪽) ──────────────────────────────────────
        panel = np.full((fh,PANEL_W,3),(18,18,18),dtype=np.uint8)
        py    = 0

        def section(title,gap=8):
            global py
            py+=gap; divider(panel,py); py+=2
            txt(panel,title,14,py+15,(110,110,110),0.40); py+=20

        py=12
        txt(panel,"FATIGUE MONITOR",14,py+18,(90,200,255),0.58,2); py+=20
        txt(panel,f"{int(elapsed//60):02d}:{int(elapsed%60):02d}",14,py+14,(90,90,90),0.40); py+=16

        section("FATIGUE INDEX",gap=6)
        cv2.rectangle(panel,(14,py),(PANEL_W-14,py+38),(28,28,28),-1)
        cv2.rectangle(panel,(14,py),(18,py+38),fi_col,-1)
        txt(panel,fi_label,28,py+16,fi_col,0.62,2)
        txt(panel,f"{fatigue_index*100:.1f}%",28,py+34,fi_col,0.48); py+=44
        draw_bar(panel,14,py,PANEL_W-28,8,fatigue_index,1.0,fi_col); py+=14

        section("EAR")
        ec=(0,200,80) if not is_closed else (60,60,220)
        txt(panel,f"Current {ear:.3f}  Thresh {personal_threshold:.3f}",14,py+16,(210,210,210),0.42); py+=20
        draw_bar(panel,14,py,PANEL_W-28,8,ear,ear_open_baseline,ec); py+=14

        section("BLINK STATS")
        txt(panel,f"Total {blink_total}  BPM {bpm:.1f}",14,py+16); py+=20
        ic=(0,165,255) if inc_ratio>0.3 else (160,160,160)
        txt(panel,f"Incomplete {incomplete_total} ({inc_ratio*100:.0f}%)  Micro {micro_total}",14,py+16,ic,0.42); py+=20

        section("PERCLOS")
        pc=(60,60,220) if perclos>PERCLOS_THRESHOLD else (0,200,80)
        txt(panel,f"{perclos*100:.1f}%  (alert>{PERCLOS_THRESHOLD*100:.0f}%)",14,py+16,pc,0.44); py+=20
        draw_bar(panel,14,py,PANEL_W-28,8,perclos,0.30,pc)
        tx=14+int(PERCLOS_THRESHOLD/0.30*(PANEL_W-28))
        cv2.line(panel,(tx,py-3),(tx,py+11),(180,60,60),2); py+=14

        section("MAR / YAWN")
        mc=(60,60,220) if mar>MAR_YAWN_THRESH else (0,200,80)
        txt(panel,f"MAR {mar:.3f}  Yawns {yawn_total} ({yawn_rpm:.1f}/min)",14,py+16,mc,0.42); py+=20
        draw_bar(panel,14,py,PANEL_W-28,8,mar,1.0,mc); py+=14
        if in_yawn: txt(panel,"YAWNING",14,py+14,(60,60,220),0.50,2); py+=18

        section("HEAD POSE")
        hc=(60,60,220) if is_drooping else (0,200,80)
        txt(panel,f"Pitch {pitch:+.1f}  Yaw {yaw_angle:+.1f}  Roll {roll:+.1f}",14,py+16,hc,0.42); py+=20
        draw_bar(panel,14,py,PANEL_W-28,8,head_droop_ratio,0.5,hc); py+=14

        section("STAGE [Rule]",gap=6)
        sc=STAGE_COLORS[blink_stage]
        cv2.rectangle(panel,(14,py),(PANEL_W-14,py+24),(30,30,30),-1)
        cv2.rectangle(panel,(14,py),(18,py+24),sc,-1)
        txt(panel,blink_stage.upper(),28,py+18,sc,0.56,2); py+=30

        if model_ready and py+40<fh-30:
            section("STAGE [AI]",gap=6)
            asc=STAGE_COLORS[ai_stage]
            cv2.rectangle(panel,(14,py),(PANEL_W-14,py+24),(30,30,30),-1)
            cv2.rectangle(panel,(14,py),(18,py+24),asc,-1)
            txt(panel,ai_stage.upper(),28,py+18,asc,0.56,2); py+=30
        elif not model_ready:
            txt(panel,training_msg or "T=train",14,py+16,(100,100,100),0.40)

        txt(panel,f"Rows:{len(collected_rows)}  S T L Q",14,fh-14,(70,70,70),0.36)
        overlay[:,:PANEL_W]=panel

        # 경고 배너
        warnings=[]
        if fi_label=="DANGER":    warnings.append(("HIGH FATIGUE DETECTED",(60,40,220)))
        elif fi_label=="CAUTION": warnings.append(("Fatigue increasing",   (60,100,210)))
        if perclos>PERCLOS_THRESHOLD: warnings.append(("PERCLOS alert!",   (60,60,200)))
        if yawn_rpm>2:            warnings.append(("Yawning frequently!",  (60,80,210)))
        if is_drooping:           warnings.append(("Head drooping",        (60,80,200)))
        if inc_ratio>0.5 and blink_total>5: warnings.append(("Too many incomplete blinks",(60,100,210)))
        for i,(msg,col) in enumerate(warnings):
            yw=fh-16-i*32
            if yw<fh//2: break
            cv2.rectangle(overlay,(PANEL_W+10,yw-20),(fw-10,yw+8),(28,28,28),-1)
            cv2.rectangle(overlay,(PANEL_W+10,yw-20),(PANEL_W+15,yw+8),col,-1)
            cv2.putText(overlay,msg,(PANEL_W+22,yw),cv2.FONT_HERSHEY_SIMPLEX,0.50,(210,210,210),1,cv2.LINE_AA)

    else:
        # 얼굴 미감지 시
        overlay[:,:PANEL_W]=(18,18,18)
        cv2.putText(overlay,"No face detected",
                    (PANEL_W+80,fh//2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(150,150,180),2,cv2.LINE_AA)
        # 별도 창: 눈 감은 채로 표시
        avatar_canvas = np.zeros((520, 520, 3), dtype=np.uint8)
        draw_avatar(avatar_canvas, 260, 260,
                    ear_ratio=0.0, mar_val=0.0, roll_deg=0.0,
                    fi_label="NORMAL", fi_col=(0,200,80), is_drooping=False)
        cv2.imshow("Claude", _avatar_canvas)

    cv2.imshow("Fatigue Detector v5", overlay)
    key = cv2.waitKey(1) & 0xFF
    if not _is_tty and key!=255:
        ch=chr(key).lower() if key<128 else ''
        if ch: _key_queue.append(ch)
    if cv2.getWindowProperty("Fatigue Detector v5", cv2.WND_PROP_VISIBLE) < 1:
        break

# 종료
_stop_input=True
cap.release()
cv2.destroyAllWindows()
detector.close()
_input_thread.join(timeout=0.3)
print("Done.")
