# calibrate.py
import numpy as np
import os

BASE_DIR = os.environ.get("FACEID_DATA_DIR", "data")

template  = np.load(os.path.join(BASE_DIR, "user_template_A.npy"))
positives = np.load(os.path.join(BASE_DIR, "positives_A.npy"))
negatives = np.load(os.path.join(BASE_DIR, "negatives_A.npy"))


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#similarities of positive
pos_sims = np.array([cos(template, e) for e in positives])
# similarities of negatives
neg_sims = np.array([cos(template, e) for e in negatives])

best_tau = None
best_stats = None

# Sweep candidate thresholds (tau) over a range to find a good decision boundary
for tau in np.linspace(0.2, 0.8, 300):

    # FAR (False Acceptance Rate): fraction of impostor (negative) similarities that would be ACCEPTED at this tau
    FAR = np.mean(neg_sims >= tau)

    # FRR (False Rejection Rate): fraction of genuine (positive) similarities that would be REJECTED at this tau
    FRR = np.mean(pos_sims < tau)

    # Choose the smallest tau that keeps security high: FAR <= 1%
    # (then we accept the corresponding FRR as the trade-off)
    if FAR <= 0.01:
        best_tau = tau
        best_stats = (FAR, FRR)
        break


print("Positives sims: mean=", pos_sims.mean(), "min=", pos_sims.min(), "max=", pos_sims.max())
print("Negatives sims: mean=", neg_sims.mean(), "min=", neg_sims.min(), "max=", neg_sims.max())
print("----")
if best_tau is not None:
    FAR, FRR = best_stats
    print(f"TAU recommended: {best_tau:.3f}")
    print(f"FAR={FAR:.3f}, FRR={FRR:.3f}")
else:
    print("More date need to be taken.")
