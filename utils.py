# utils.py
import os, glob, cv2, numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

OUT_SIZE = 112

# Guide template for alignment
ARC_TEMPLATE = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32)

_app = FaceAnalysis(name="buffalo_l")
_app.prepare(ctx_id=0, det_size=(640, 640))  # CPU

def _find_arcface():
    cand = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
    if os.path.exists(cand): return get_model(cand)
    home = os.path.expanduser("~/.insightface/models")
    matches = glob.glob(os.path.join(home, "**", "w600k_r50.onnx"), recursive=True)
    if matches: return get_model(matches[0])
    raise RuntimeError("w600k_r50.onnx not found in ~/.insightface/models")

_arc = _find_arcface()
_arc.prepare(ctx_id=0)

def _estimate_similarity(src5: np.ndarray, dst5: np.ndarray):
    """Returns a matriz 2x3 of similarity using least squares."""
    M, _ = cv2.estimateAffinePartial2D(src5, dst5, method=cv2.LMEDS)
    return M  # 2x3

def _align_5pt(img_bgr):
    """Detection, take biggest face,align with the fixed matrix """
    H, W = img_bgr.shape[:2]
    faces = _app.get(img_bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    if f.kps is None or len(f.kps) != 5:
        return None
    src = f.kps.astype(np.float32)               # 5x2: LE, RE, Nose, LM, RM
    dst = ARC_TEMPLATE.copy()
    M = _estimate_similarity(src, dst)
    if M is None:
        return None
    aligned = cv2.warpAffine(img_bgr, M, (OUT_SIZE, OUT_SIZE), flags=cv2.INTER_LINEAR)
    return aligned

def get_face_embedding(img_bgr):
    """Compute an ArcFace embedding for the given BGR image.
      Detect + align face to 112x112 using 5-point landmarks.
      Run ArcFace to get a 512-D identity embedding.
      L2-normalize the embedding (important for cosine similarity).
      emb: np.ndarray shape (512,) float32, L2-normalized
      """
    aligned = _align_5pt(img_bgr)
    if aligned is None:
        return None
    try:
        emb = _arc.get_feat(aligned)
    except AttributeError:
        emb = _arc.get(aligned)
    emb = np.asarray(emb, np.float32).reshape(-1)
    n = np.linalg.norm(emb)
    if n > 0: emb /= n
    return emb

