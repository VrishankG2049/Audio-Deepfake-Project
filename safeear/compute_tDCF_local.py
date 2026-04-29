import numpy as np

# -----------------------------
# Load CM scores
# -----------------------------
def load_scores(file):
    scores = []
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            score = float(parts[1])
            label = 1 if parts[2] == "spoof" else 0
            scores.append((score, label))
    return scores


# -----------------------------
# Compute EER
# -----------------------------
def compute_eer(scores):
    scores = sorted(scores, key=lambda x: x[0])
    labels = np.array([l for _, l in scores])

    fars = []
    frrs = []

    for i in range(len(scores)):
        thresh = scores[i][0]

        preds = np.array([1 if s >= thresh else 0 for s, _ in scores])

        fa = np.sum((preds == 1) & (labels == 0)) / np.sum(labels == 0)
        fr = np.sum((preds == 0) & (labels == 1)) / np.sum(labels == 1)

        fars.append(fa)
        frrs.append(fr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[idx] + frrs[idx]) / 2

    return eer


# -----------------------------
# Approximate t-DCF
# -----------------------------
def compute_tdcf(scores):
    scores = sorted(scores, key=lambda x: x[0])
    labels = np.array([l for _, l in scores])

    tdcf_values = []

    for i in range(len(scores)):
        thresh = scores[i][0]

        preds = np.array([1 if s >= thresh else 0 for s, _ in scores])

        # CM miss and false alarm
        P_miss_cm = np.sum((preds == 0) & (labels == 1)) / np.sum(labels == 1)
        P_fa_cm = np.sum((preds == 1) & (labels == 0)) / np.sum(labels == 0)

        # simplified t-DCF (normalized)
        tdcf = P_miss_cm + P_fa_cm
        tdcf_values.append(tdcf)

    return np.min(tdcf_values)


# -----------------------------
# MAIN
# -----------------------------
scores = load_scores("scores.txt")

eer = compute_eer(scores)
tdcf = compute_tdcf(scores)

print(f"EER: {eer:.4f}")
print(f"Approx t-DCF: {tdcf:.4f}")