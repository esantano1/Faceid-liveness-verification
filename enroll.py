import cv2, numpy as np, os
from utils import get_face_embedding

BASE = os.environ.get("FACEID_DATA_DIR", "data")
os.makedirs(BASE, exist_ok=True)
TEMPLATE_PATH  = os.path.join(BASE, "user_template_A.npy")
POSITIVES_PATH = os.path.join(BASE, "positives_A.npy")

# Camera: macOS backend if available, default backend otherwise (Linux/Windows)
cap = cv2.VideoCapture(0, getattr(cv2, "CAP_AVFOUNDATION", 0))
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")
embs = []

print("Take 30 samples")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    # Compute embedding for the current frame.
    # get_face_embedding internally:
    # detects face + 5 landmarks,
    # aligns to ArcFace canonical 112x112,
    # extracts a 512-D identity embedding (L2-normalized).

    emb = get_face_embedding(frame)
    if emb is not None:
        embs.append(emb)
        cv2.putText(frame, f"capture: {len(embs)}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("enroll", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or len(embs) >= 30:  # ESC o 30 muestras
        break

cap.release()
cv2.destroyAllWindows()

#We put all the embeddings into a matrix E
if embs:
    E = np.stack(embs)
    template = E.mean(axis=0)
    np.save(TEMPLATE_PATH, template)
    np.save(POSITIVES_PATH, E)
    print(f"[OK] Saved:\n- {TEMPLATE_PATH}\n- {POSITIVES_PATH} ({E.shape})")
else:
    print("Wrong")