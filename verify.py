# verify.py
import cv2
import numpy as np
from pathlib import Path
from utils import get_face_embedding

# ---- Paths ----
import os
BASE = Path(os.environ.get("FACEID_DATA_DIR", "data"))
TEMPLATE_PATH = BASE / "user_template_A.npy"
template = np.load(TEMPLATE_PATH).astype(np.float32)
n = np.linalg.norm(template)
if n > 0:
    template /= n

# ---- Threshold we take ----
THRESHOLD = 0.4   # Provisional: re-run calibrate.py on your own data and update it

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Here we compute the cosine similarity"""
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

cap = cv2.VideoCapture(0, getattr(cv2, "CAP_AVFOUNDATION", 0))
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    text, color = "No face", (0, 255, 255)
    emb = get_face_embedding(frame)             #Here we are taking the embedding of the live face
    if emb is not None:
        sim = cosine_sim(template, emb)         #Here we compute the cosine similarity of the template and the live face
        if sim >= THRESHOLD:
            text, color = f"ACCEPT ({sim:.3f})", (0, 255, 0)            #Accept or Reject
        else:
            text, color = f"REJECT ({sim:.3f})", (0, 0, 255)

    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("verify", frame)
    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
