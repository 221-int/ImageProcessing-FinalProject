import cv2
import numpy as np
from collections import deque
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── 눈 랜드마크 인덱스 ─────────────────────────────────────────────
# MediaPipe가 감지하는 468개 얼굴 포인트 중 왼쪽/오른쪽 눈 6개 포인트 번호
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── 주요 파라미터 설정 ────────────────────────────────────────────
CALIBRATION_FRAMES   = 60     # 개인 눈 기준값 측정에 사용할 프레임 수
SLIDING_WINDOW       = 300    # EAR 이동 평균 계산용 버퍼 크기
BLINK_CONSEC_FRAMES  = 2      # 몇 프레임 이상 눈이 감겨야 깜빡임으로 인정
PERCLOS_WINDOW       = 900    # PERCLOS 계산 대상 최근 프레임 수 (약 30초)
PERCLOS_THRESHOLD    = 0.15   # PERCLOS 15% 초과 시 졸음 경고
FATIGUE_RATIO_THRESH = 0.6    # 눈 열리는 속도 / 감기는 속도 비율 임계값
PANEL_W              = 320    # 왼쪽 정보 패널 너비(px)

# ── MediaPipe 얼굴 랜드마크 감지기 초기화 ────────────────────────
# face_landmarker.task 모델 파일을 불러와 단일 얼굴, 정지 이미지 모드로 설정
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


# ── EAR 계산 ─────────────────────────────────────────────────────
# Eye Aspect Ratio: 눈의 세로 길이 / 가로 길이 비율
# 값이 작을수록 눈이 감긴 상태 (완전히 감기면 ~0.0, 뜨면 ~0.3)
def get_3d_ear(landmarks, eye_indices, img_w, img_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w]))
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


# ── 고개 방향 보정 ────────────────────────────────────────────────
# 고개를 옆으로 돌리면 눈이 좁아 보여 EAR이 낮게 측정됨 → 보정 계수로 상쇄
def get_head_pose_correction(landmarks):
    left  = landmarks[130]
    right = landmarks[359]
    yaw   = abs(left.x - right.x)
    return 1.0 + 0.08 * (1.0 - min(yaw / 0.4, 1.0))


# ── 깜빡임 완전도 계산 ────────────────────────────────────────────
# 눈을 얼마나 완전히 감았는지 0~1 사이 값으로 반환
# (기준 EAR - 깜빡임 중 최솟값) / 기준 EAR
def compute_completeness(ear_open, ear_min):
    if ear_open < 1e-6:
        return 0.0
    return (ear_open - ear_min) / ear_open


# ── 깜빡임 등급 분류 ──────────────────────────────────────────────
# Complete(정상) / Incomplete(불완전) / Micro(미세) 로 구분 + 표시 색상 반환
def classify_completeness(c):
    if c > 0.9:   return "Complete",   (0, 200, 0)
    elif c > 0.5: return "Incomplete", (0, 165, 255)
    else:         return "Micro",      (0, 0, 255)


# ── UI 헬퍼 함수들 ────────────────────────────────────────────────
# draw_bar : 패널에 진행률 막대 그리기 (EAR, PERCLOS 시각화에 사용)
# txt      : 이미지에 텍스트 출력 단축 함수
# divider  : 패널 섹션 사이 구분선 그리기
def draw_bar(img, x, y, w, h, value, max_val, color_fill, color_bg=(50,50,50)):
    cv2.rectangle(img, (x, y), (x+w, y+h), color_bg, -1)
    fill = int(min(value / max(max_val, 1e-6), 1.0) * w)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+h), color_fill, -1)

def txt(img, text, x, y, color=(210,210,210), scale=0.50, bold=1):
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

def divider(img, y):
    cv2.line(img, (14, y), (PANEL_W-14, y), (50,50,50), 1)


# ── 상태 변수 초기화 ──────────────────────────────────────────────
# 캘리브레이션, 깜빡임 감지, 각종 누적 통계에 사용하는 전역 상태값
calib_ears         = []                          # 캘리브레이션 중 수집한 EAR 값들
personal_threshold = None                        # 개인화된 눈 감김 판단 임계값
ear_open_baseline  = None                        # 캘리브레이션으로 구한 눈 뜬 기준 EAR
calibrated         = False
ear_window         = deque(maxlen=SLIDING_WINDOW)    # 최근 EAR 이력 (드리프트 보정용)
eye_closed_log     = deque(maxlen=PERCLOS_WINDOW)    # 눈 감김 여부 이력 (PERCLOS 계산용)
closing_slopes     = deque(maxlen=20)                # 눈 감기는 속도 이력
opening_slopes     = deque(maxlen=20)                # 눈 열리는 속도 이력
blink_total        = 0
incomplete_total   = 0
micro_total        = 0
blink_consec       = 0      # 현재 연속으로 눈 감긴 프레임 수
in_blink           = False  # 현재 깜빡임 진행 중 여부
blink_ear_min      = 1.0    # 현재 깜빡임 중 가장 낮은 EAR (완전도 계산용)
prev_ear           = None
start_time         = time.time()

# ── 카메라 초기화 ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Started. Press Q or ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
    fh, fw  = frame.shape[:2]
    elapsed = time.time() - start_time
    overlay = frame.copy()

    # BGR → RGB 변환 후 MediaPipe에 전달해 얼굴 랜드마크 감지
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpi   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res   = detector.detect(mpi)

    if res.face_landmarks:
        lms = res.face_landmarks[0]

        # 양쪽 EAR 평균 + 고개 방향 보정 적용
        le  = get_3d_ear(lms, LEFT_EYE,  fw, fh)
        re  = get_3d_ear(lms, RIGHT_EYE, fw, fh)
        cor = get_head_pose_correction(lms)
        ear = ((le + re) / 2.0) * cor
        ear_window.append(ear)

        # ── 캘리브레이션 단계 ─────────────────────────────────────
        # 처음 60프레임 동안 EAR을 수집해 개인 기준값과 임계값 산출
        # 완료되면 calibrated = True로 전환하고 메인 루프로 진입
        if not calibrated:
            calib_ears.append(ear)
            prog = len(calib_ears) / CALIBRATION_FRAMES
            bw   = int(480 * prog)
            cx, cy = fw//2, fh//2
            cv2.rectangle(overlay, (cx-240,cy-40), (cx+240,cy+40), (30,30,30), -1)
            cv2.rectangle(overlay, (cx-220,cy-14), (cx+220,cy+14), (55,55,55), -1)
            if bw > 0:
                cv2.rectangle(overlay, (cx-220,cy-14), (cx-220+bw,cy+14), (0,180,90), -1)
            cv2.putText(overlay, f"Calibrating... {int(prog*100)}%",
                        (cx-130, cy-22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (200,200,200), 2, cv2.LINE_AA)
            if len(calib_ears) >= CALIBRATION_FRAMES:
                ear_open_baseline  = np.percentile(calib_ears, 80)   # 상위 80% → 눈 뜬 기준
                personal_threshold = ear_open_baseline * 0.75        # 기준의 75% → 감김 판단선
                calibrated         = True
                print(f"Calibration done  threshold={personal_threshold:.3f}")
            cv2.imshow("Blink Detector", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            continue

        # ── 드리프트 보정 ─────────────────────────────────────────
        # 시간이 지나면서 EAR 기준값이 서서히 변할 수 있으므로
        # 최근 EAR 이력의 80 percentile로 임계값을 주기적으로 재조정
        if len(ear_window) >= 30:
            ro = np.percentile(list(ear_window), 80)
            if ro > ear_open_baseline * 0.5:
                personal_threshold = ro * 0.75

        # ── 깜빡임 감지 로직 ──────────────────────────────────────
        # EAR이 임계값 미만이면 눈 감김으로 판단
        # 눈이 다시 열릴 때 연속 프레임 수가 기준 이상이면 깜빡임 1회 카운트
        # 깜빡임의 완전도를 계산해 Complete / Incomplete / Micro 분류
        is_closed = ear < personal_threshold
        if prev_ear is not None:
            slope = ear - prev_ear
            if slope < -0.005:  closing_slopes.append(abs(slope))  # 감기는 속도
            elif slope > 0.005: opening_slopes.append(abs(slope))  # 열리는 속도

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

        # ── 피로도 지표 계산 ──────────────────────────────────────
        # PERCLOS  : 최근 900프레임 중 눈 감긴 비율 → 졸음 지표
        # fatigue_ratio : 눈 열리는 속도 / 감기는 속도 → 1보다 작으면 피로
        # bpm      : 분당 깜빡임 횟수 (정상 12~20회)
        # inc_ratio: 전체 깜빡임 중 불완전 깜빡임 비율
        perclos       = sum(eye_closed_log) / max(len(eye_closed_log), 1)
        avg_c         = np.mean(closing_slopes) if closing_slopes else 0.01
        avg_o         = np.mean(opening_slopes) if opening_slopes else 0.01
        fatigue_ratio = avg_o / (avg_c + 1e-6)
        bpm           = blink_total / max(elapsed / 60.0, 1/60.0)
        inc_ratio     = incomplete_total / max(blink_total, 1)

        # 눈 랜드마크 포인트를 화면에 점으로 표시
        for idx in LEFT_EYE + RIGHT_EYE:
            lm = lms[idx]
            cv2.circle(overlay, (int(lm.x*fw), int(lm.y*fh)), 2, (0,230,180), -1)

        # ── 왼쪽 정보 패널 렌더링 ────────────────────────────────
        # EAR / 깜빡임 통계 / PERCLOS / 피로도 / 눈 상태를 섹션별로 표시
        panel = np.full((fh, PANEL_W, 3), (18,18,18), dtype=np.uint8)

        txt(panel, "BLINK MONITOR", 14, 34, (90,200,255), 0.62, 2)
        txt(panel, f"{int(elapsed//60):02d}:{int(elapsed%60):02d}",
            14, 54, (90,90,90), 0.42)

        divider(panel, 66)
        txt(panel, "EAR", 14, 84, (110,110,110), 0.42)
        txt(panel, f"Current    {ear:.3f}", 14, 106)
        txt(panel, f"Threshold  {personal_threshold:.3f}", 14, 126, (150,150,150), 0.46)
        ear_col = (0,200,80) if not is_closed else (60,60,220)
        draw_bar(panel, 14, 136, PANEL_W-28, 10, ear, ear_open_baseline, ear_col)

        divider(panel, 158)
        txt(panel, "BLINK STATS", 14, 176, (110,110,110), 0.42)
        txt(panel, f"Total   {blink_total}", 14, 200)
        bpm_col = (0,200,80) if 12 <= bpm <= 20 else (60,100,230)
        txt(panel, f"BPM     {bpm:.1f}", 14, 222, bpm_col)
        inc_col = (0,165,255) if inc_ratio > 0.3 else (160,160,160)
        txt(panel, f"Incomplete  {incomplete_total}  ({inc_ratio*100:.0f}%)", 14, 244, inc_col)
        txt(panel, f"Micro   {micro_total}", 14, 264, (140,140,140))

        divider(panel, 280)
        txt(panel, "PERCLOS", 14, 298, (110,110,110), 0.42)
        pc_col = (60,60,220) if perclos > PERCLOS_THRESHOLD else (0,200,80)
        txt(panel, f"{perclos*100:.1f}%", 14, 322, pc_col, 0.62, 2)
        txt(panel, f"alert > {PERCLOS_THRESHOLD*100:.0f}%", 14, 342, (90,90,90), 0.42)
        draw_bar(panel, 14, 352, PANEL_W-28, 10, perclos, 0.30, pc_col)
        tx = 14 + int(PERCLOS_THRESHOLD / 0.30 * (PANEL_W-28))
        cv2.line(panel, (tx, 348), (tx, 366), (180,60,60), 2)

        divider(panel, 378)
        txt(panel, "FATIGUE", 14, 396, (110,110,110), 0.42)
        fr_col = (60,60,220) if fatigue_ratio < FATIGUE_RATIO_THRESH else (0,200,80)
        txt(panel, f"Open/Close  {fatigue_ratio:.2f}", 14, 420, fr_col)
        txt(panel, f"alert < {FATIGUE_RATIO_THRESH}", 14, 440, (90,90,90), 0.42)

        divider(panel, 456)
        txt(panel, "STATUS", 14, 474, (110,110,110), 0.42)
        if is_closed:
            cv2.rectangle(panel, (14,484), (PANEL_W-14,514), (35,15,15), -1)
            cv2.rectangle(panel, (14,484), (18,514), (60,60,220), -1)
            txt(panel, "EYE CLOSED", 28, 506, (80,80,230), 0.60, 2)
        else:
            cv2.rectangle(panel, (14,484), (PANEL_W-14,514), (15,35,15), -1)
            cv2.rectangle(panel, (14,484), (18,514), (0,200,80), -1)
            txt(panel, "EYE OPEN", 28, 506, (0,200,80), 0.60, 2)

        overlay[:, :PANEL_W] = panel

        # ── 경고 메시지 표시 ──────────────────────────────────────
        # PERCLOS 초과 / 피로도 감지 / 낮은 깜빡임 빈도 / 불완전 깜빡임 과다
        # 해당하는 항목을 화면 하단에 배너로 표시
        warnings = []
        if perclos > PERCLOS_THRESHOLD:
            warnings.append(("PERCLOS alert: blink more!", (60,60,210)))
        if fatigue_ratio < FATIGUE_RATIO_THRESH and blink_total > 5:
            warnings.append(("Eye fatigue detected", (60,100,220)))
        if bpm < 8 and elapsed > 30:
            warnings.append(("Low blink rate!", (60,140,210)))
        if inc_ratio > 0.5 and blink_total > 5:
            warnings.append(("Too many incomplete blinks", (60,120,210)))

        for i, (msg, col) in enumerate(warnings):
            yw = fh - 16 - i*34
            cv2.rectangle(overlay, (PANEL_W+10,yw-22), (fw-10,yw+10), (28,28,28), -1)
            cv2.rectangle(overlay, (PANEL_W+10,yw-22), (PANEL_W+15,yw+10), col, -1)
            cv2.putText(overlay, msg, (PANEL_W+22, yw),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210,210,210), 1, cv2.LINE_AA)

    else:
        # 얼굴이 감지되지 않으면 패널을 어둡게 하고 안내 문구 표시
        overlay[:, :PANEL_W] = (18,18,18)
        cv2.putText(overlay, "No face detected",
                    (PANEL_W + (fw-PANEL_W)//2 - 140, fh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,180), 2, cv2.LINE_AA)