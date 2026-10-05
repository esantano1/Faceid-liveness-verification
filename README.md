# Face ID: Face Verification with Liveness Detection

A face verification system in Python that mimics the core idea behind phone "Face ID" unlock. It enrolls a user, calibrates a decision threshold against impostor samples, and verifies a live camera feed, with an optional **blink-based liveness check** to reject static photos.

Developed as a team course project (4 students) at Purdue University (ECE).

> **My contribution:** I built the data-collection part of the pipeline: the enrollment script (`enroll.py`, genuine samples and user template) and the impostor collection script (`collect_negatives.py`).

## How it works

The system follows four steps:

1. **Enrollment** (`enroll.py`): captures ~30 frames of the user, computes a 512-D identity embedding for each one, and stores their mean as the user template.
2. **Negatives** (`collect_negatives.py`): collects embeddings of *other* people to model impostors.
3. **Calibration** (`calibrate.py`): computes cosine similarities of genuine and impostor samples against the template and sweeps a threshold τ. It reports FAR (False Accept Rate) and FRR (False Reject Rate) and picks the lowest τ with FAR ≤ 1%.
4. **Live verification** (`verify.py`, `verify_blink.py`): embeds the live face, computes `cos(template, embedding)` and accepts if it is ≥ τ. `verify_blink.py` adds liveness: the face is only checked after a blink is detected.

### Face pipeline (`utils.py`)

- **Detection + 5 landmarks** (eyes, nose, mouth corners) with InsightFace `buffalo_l`.
- **Alignment:** a similarity transform maps the landmarks onto the canonical ArcFace 112×112 template.
- **Embedding:** ArcFace (`w600k_r50`, ONNX) produces a 512-D vector, L2-normalised so cosine similarity becomes a dot product.

### Liveness (`verify_blink.py`)

MediaPipe Face Mesh provides eye landmarks. The **Eye Aspect Ratio (EAR)** is computed for both eyes; if it stays below 0.22 for at least 2 consecutive frames, a blink is registered. Verification only runs if a blink occurred in the last 10 seconds.

## Project structure

```
utils.py              Detection, alignment and ArcFace embedding
enroll.py             Build the user template (~30 samples)
collect_negatives.py  Collect impostor embeddings
calibrate.py          Sweep threshold τ, report FAR / FRR
verify.py             Live verification (no liveness)
verify_blink.py       Live verification + blink liveness
data/                 Your local embeddings (git-ignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

InsightFace downloads the `buffalo_l` models to `~/.insightface/models` on first run.

By default, scripts read and write embeddings in `./data`. To use another folder:

```bash
export FACEID_DATA_DIR=/path/to/your/data
```

## Usage

```bash
python enroll.py             # 1. capture ~30 samples of the user (ESC to stop)
python collect_negatives.py  # 2. SPACE to save a sample of another person, ESC to quit
python calibrate.py          # 3. prints the recommended threshold τ, FAR and FRR
python verify.py             # 4a. live verification
python verify_blink.py       # 4b. live verification + liveness (blink)
```

After calibrating, set the `THRESHOLD` constant at the top of `verify.py` and `verify_blink.py` (keep both with the same value).

> The scripts use the macOS camera backend when available and fall back to the default one, so they work on macOS, Linux and Windows. On Windows, installing `insightface` may require the Microsoft C++ Build Tools.

## Privacy

Face embeddings are biometric data. **This repository does not include any embeddings, photos or videos**, and `.gitignore` blocks `.npy` files and images. Collect data only from people who have given you their consent.

## Limitations

- **Small, informal calibration set.** Thresholds come from ~30 genuine and ~100 impostor samples captured by the team, not from a formal evaluation, so the FAR/FRR numbers should not be read as benchmark results.
- **Blink liveness can be bypassed.** A video of the person blinking could pass the check. It blocks photos and simple screen attacks, not replay attacks or 3D masks.
- **Sensitive to conditions.** Poor lighting, blur, glasses and strong front light affect the results.
- **Provisional threshold.** Both verification scripts ship with a provisional `THRESHOLD = 0.4`; re-run `calibrate.py` on your own data (ideally with separate live test samples) and update it.
- **Single user.** One template per run; no multi-user database.

## Possible improvements

- Stronger anti-spoofing (depth, texture analysis or a challenge-response check).
- Proper evaluation with a held-out test set (ROC curve, EER).
- Multi-user enrollment and a config file instead of hard-coded constants.

## Tech stack

Python · OpenCV · InsightFace (ArcFace, `buffalo_l`) · ONNX Runtime · MediaPipe Face Mesh · NumPy
