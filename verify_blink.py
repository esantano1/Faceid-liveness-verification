# verify_blink.py
import cv2
import numpy as np
import os, time, collections
import mediapipe as mp
from utils import get_face_embedding

# ====== Paths ======
BASE = os.environ.get("FACEID_DATA_DIR", "data")
TEMPLATE_PATH = os.path.join(BASE, "user_template_A.npy")

# ====== THRESHOLD (τ)  ======
THRESHOLD = 0.4  # Provisional: re-run calibrate.py on your own data and update it

# ====== LIVENESS ======
BLINK_WINDOW_SEC = 10.0     #At least 1 blink in 10 s
EAR_THR = 0.22              # EAR, if it is behind this value the eye is closed
EAR_CONSEC_FRAMES = 2       # Number of frames the eye has to be closed


template = np.load(TEMPLATE_PATH).astype(np.float32)
template = template.reshape(-1)
n = np.linalg.norm(template)
if n > 0:
    template /= n

def cos(a, b):
    """
    Cosine similarity between two vectors.
    With L2-normalized vectors, this becomes a dot product in [-1, 1].
    Returns 0 if either vector has zero norm.
    """
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,                #only track the closest face
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Indices of 6 landmarks per eye used for EAR:
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]   # [outer, upper1, upper2, inner, lower2, lower1]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

def euclid(p1, p2):
    """Euclidean distance between 2D points"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(pts):
    """EAR function
    it is high when the eye is open and low when it is closed
    """
    p1, p2, p3, p4, p5, p6 = pts
    v1 = euclid(p2, p6)
    v2 = euclid(p3, p5)
    h  = euclid(p1, p4)
    return 0.0 if h == 0 else (v1 + v2) / (2.0 * h)

# Control of blinks
blink_times = collections.deque()

def blink_in_last(seconds):
    """
    Returns True if at least one blink happened within the last `seconds`.
    Also removes old blink timestamps outside the window
    """
    now = time.time()
    while blink_times and now - blink_times[0] > seconds:
        blink_times.popleft()
    return len(blink_times) > 0

state = {'closed_frames': 0}

def update_blinks(earL, earR):
    ear = (earL + earR) / 2.0
    closed = ear < EAR_THR
    if closed:
        state['closed_frames'] += 1
    else:
        if state['closed_frames'] >= EAR_CONSEC_FRAMES:
            blink_times.append(time.time())
        state['closed_frames'] = 0

# ====== Camera ======
cap = cv2.VideoCapture(0, getattr(cv2, "CAP_AVFOUNDATION", 0))
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

print("Liveness")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    earL = earR = 0.0
    has_face = False
    blink_text = "No blink"
    verify_text = "Waiting blink..."
    color = (0, 255, 255)

    #Liveness (blink)
    # If finds a face, we compute the EAR from both eyes
    if res.multi_face_landmarks:
        has_face = True
        lm = res.multi_face_landmarks[0].landmark

        def get_pts(idxs):
            return [(lm[i].x * w, lm[i].y * h) for i in idxs]

        left_pts  = get_pts(LEFT_EYE_IDX)
        right_pts = get_pts(RIGHT_EYE_IDX)
        earL = eye_aspect_ratio(left_pts)
        earR = eye_aspect_ratio(right_pts)

        update_blinks(earL, earR)
        if blink_in_last(BLINK_WINDOW_SEC):
            blink_text = "Blink OK (live)"

    # If we had a recent blink we compute the live verification
    if has_face and blink_in_last(BLINK_WINDOW_SEC):
        emb = get_face_embedding(frame)             #Take the embedding of the live frame
        if emb is not None:
            emb = np.asarray(emb).reshape(-1)  # fuerza 1D (512,)
            sim = cos(template, emb)                #We compute the cosine with the template of the enroll
            if sim >= THRESHOLD:                    #If it is greater than the threshold we accept
                verify_text = f"ACCEPT ({sim:.3f})"
                color = (0, 200, 0)
            else:
                verify_text = f"REJECT ({sim:.3f})"
                color = (0, 0, 255)
        else:
            verify_text = "Face not detected"
            color = (0, 255, 255)
    else:
        if not has_face:
            verify_text = "No face"
        elif not blink_in_last(BLINK_WINDOW_SEC):
            verify_text = "Waiting blink..."

    # Overlays
    cv2.putText(frame, f"EAR L/R: {earL:.2f}/{earR:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, blink_text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, verify_text, (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("verify_blink", frame)
    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
