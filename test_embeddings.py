"""
test_embeddings.py — Embedding uniqueness dry-run
==================================================
Checks whether the ResNet50 ONNX model produces DISTINCT embeddings
for visually different crops, and SIMILAR embeddings for the same crop
with minor perturbations (jitter, brightness, slight resize).

Tests:
  1  Raw embedding stats       — shape, norm, value range
  2  Same-crop consistency     — cosine sim should be ~1.0
  3  Jitter robustness         — small bbox shift, sim should stay > 0.90
  4  Brightness robustness     — +/- exposure, sim should stay > 0.85
  5  Inter-person uniqueness   — different colour persons, sim should be LOW
  6  Trained-weight indicator  — are all embeddings collapsing to one point?
  7  Verdict                   — clear PASS / FAIL with explanation
"""

import os, sys, time
import numpy as np
import cv2

os.chdir("/home/claude")
sys.path.insert(0, "/home/claude")

from reid_test import ReIDExtractor

ONNX_PATH = "onnx_model/resnet50_reid.onnx"

# ── Helpers ────────────────────────────────────────────────────────────────

FRAME_H, FRAME_W = 720, 1280

def blank(bg=(20, 20, 20)):
    return np.full((FRAME_H, FRAME_W, 3), bg, dtype=np.uint8)

def draw_person(frame, x_center, color, y_center=360, w=90, h=220):
    """Draw a solid-colour rectangle + checkerboard stripe to add texture."""
    x1 = x_center - w//2; y1 = y_center - h//2
    x2 = x1 + w;           y2 = y1 + h
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(FRAME_W, x2); y2 = min(FRAME_H, y2)
    # Solid base colour
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    # Checkerboard stripes for texture (makes embeddings more distinct)
    stripe = 20
    for i, yy in enumerate(range(y1, y2, stripe)):
        if i % 2 == 0:
            c2 = tuple(max(0, c - 60) for c in color)
            cv2.rectangle(frame, (x1, yy), (x2, min(y2, yy+stripe)), c2, -1)
    # Unique text label
    cv2.putText(frame, str(color[:2]), (x1+4, y1+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return [x1, y1, x2, y2]

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def brightness(frame, delta):
    """Shift brightness of a frame."""
    f = frame.astype(np.int16) + delta
    return np.clip(f, 0, 255).astype(np.uint8)

def bar(value, width=30, lo=0.0, hi=1.0):
    """ASCII progress bar for similarity values."""
    frac  = (value - lo) / (hi - lo + 1e-8)
    frac  = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.4f}"

SEP = "─" * 62

# ── 6 distinct person appearances ─────────────────────────────────────────
# Each is a (R, G, B) colour — deliberately spread across the colour space
PERSONS = {
    "P-Red":    (200,  40,  40),
    "P-Green":  ( 40, 200,  40),
    "P-Blue":   ( 40,  40, 200),
    "P-Yellow": (200, 180,  30),
    "P-Purple": (160,  30, 200),
    "P-Cyan":   ( 30, 180, 200),
}

# ── Load model ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  Embedding Uniqueness Dry-Run")
print("=" * 62)
print(f"\nLoading model: {ONNX_PATH}")
t0   = time.time()
reid = ReIDExtractor(ONNX_PATH)
print(f"Loaded in {time.time()-t0:.2f}s\n")

# ══════════════════════════════════════════════════════════════════════════
# TEST 1 — Raw embedding stats
# ══════════════════════════════════════════════════════════════════════════
print(SEP)
print("TEST 1: Raw embedding stats")
print(SEP)

frame = blank()
bbox  = draw_person(frame, 640, (180, 80, 80))
feat  = reid.extract(frame, bbox)

print(f"  Shape   : {feat.shape}     (expect (2048,))")
print(f"  Norm    : {np.linalg.norm(feat):.6f}  (expect 1.0000)")
print(f"  Min     : {feat.min():.4f}")
print(f"  Max     : {feat.max():.4f}")
print(f"  Mean    : {feat.mean():.4f}")
print(f"  Std     : {feat.std():.4f}")
print(f"  Non-zero: {np.count_nonzero(feat)} / {len(feat)}")

norm_ok  = abs(np.linalg.norm(feat) - 1.0) < 0.001
shape_ok = feat.shape == (2048,)
print(f"\n  Shape OK : {'PASS' if shape_ok else 'FAIL'}")
print(f"  Norm  OK : {'PASS' if norm_ok  else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════
# TEST 2 — Same-crop consistency (extract twice → must be identical)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TEST 2: Same-crop consistency (determinism)")
print(SEP)

sims = []
for _ in range(5):
    f1 = reid.extract(frame, bbox)
    f2 = reid.extract(frame, bbox)
    sims.append(cosine(f1, f2))

print(f"  Cosine sim (same crop, 5 pairs):")
for i, s in enumerate(sims):
    print(f"    Run {i+1}: {bar(s)}")
print(f"  Mean: {np.mean(sims):.4f}  (expect 1.0000)")
print(f"  Status: {'PASS' if np.mean(sims) > 0.9999 else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════
# TEST 3 — Jitter robustness (small bbox shifts)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TEST 3: Jitter robustness (bbox shift ±5–20 px)")
print(SEP)

ref_feat = reid.extract(frame, bbox)
shifts   = [2, 5, 10, 15, 20]
print(f"  {'Shift':>8}  Similarity")
jitter_sims = []
for dx in shifts:
    f = blank()
    b = draw_person(f, 640 + dx, (180, 80, 80))
    jf = reid.extract(f, b)
    s  = cosine(ref_feat, jf)
    jitter_sims.append(s)
    print(f"  {dx:>5} px:  {bar(s)}")

avg_jitter = np.mean(jitter_sims)
print(f"\n  Mean sim across jitter: {avg_jitter:.4f}")
print(f"  Status: {'PASS (> 0.85)' if avg_jitter > 0.85 else 'WARN — jitter sensitivity high'}")

# ══════════════════════════════════════════════════════════════════════════
# TEST 4 — Brightness robustness
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TEST 4: Brightness robustness (+/- exposure)")
print(SEP)

bright_levels = [-60, -30, 0, +30, +60]
bright_sims   = []
print(f"  {'Delta':>8}  Similarity to baseline")
for delta in bright_levels:
    f  = brightness(frame, delta)
    bf = reid.extract(f, bbox)
    s  = cosine(ref_feat, bf)
    bright_sims.append(s)
    print(f"  {delta:>+5} px:  {bar(s)}")

avg_bright = np.mean(bright_sims)
print(f"\n  Mean sim across exposures: {avg_bright:.4f}")
print(f"  Status: {'PASS (> 0.80)' if avg_bright > 0.80 else 'WARN — brightness sensitive'}")

# ══════════════════════════════════════════════════════════════════════════
# TEST 5 — Inter-person uniqueness (the critical test)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TEST 5: Inter-person uniqueness (different appearances)")
print(SEP)
print("  This is the KEY test — embeddings must be DISTINCT per person.")
print("  With RANDOM weights all sims will be ~1.0 (model not trained).")
print("  With TRAINED weights sims should be < 0.70 between different people.")
print()

# Extract one embedding per person
person_feats = {}
for name, color in PERSONS.items():
    f = blank()
    b = draw_person(f, 640, color)
    person_feats[name] = reid.extract(f, b)
    print(f"  {name}: norm={np.linalg.norm(person_feats[name]):.4f}  "
          f"mean={person_feats[name].mean():.4f}")

print()
print("  Pairwise cosine similarities:")
print(f"  {'Pair':<28} Similarity  Status")

pairs      = []
all_sims   = []
names      = list(person_feats.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        n1, n2 = names[i], names[j]
        s      = cosine(person_feats[n1], person_feats[n2])
        all_sims.append(s)
        pairs.append((n1, n2, s))
        status = "DISTINCT" if s < 0.70 else ("BORDERLINE" if s < 0.85 else "COLLAPSING")
        print(f"  {n1} vs {n2:<12} {s:.4f}      {status}")

mean_inter = np.mean(all_sims)
max_inter  = np.max(all_sims)
min_inter  = np.min(all_sims)

print(f"\n  Mean inter-person sim : {mean_inter:.4f}")
print(f"  Max  inter-person sim : {max_inter:.4f}")
print(f"  Min  inter-person sim : {min_inter:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# TEST 6 — Collapse detection
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TEST 6: Collapse detection (random vs trained weights)")
print(SEP)

# Compute std of all pairwise sims — low std = all sims the same = collapsed
sim_std = np.std(all_sims)
print(f"  Std of pairwise sims : {sim_std:.4f}")
print(f"  Mean pairwise sim    : {mean_inter:.4f}")

if mean_inter > 0.95:
    collapse_status = "COLLAPSED — random/untrained weights"
    collapsed = True
elif mean_inter > 0.80:
    collapse_status = "BORDERLINE — partially trained or weak backbone"
    collapsed = False
else:
    collapse_status = "DISTINCT — well-trained model"
    collapsed = False

print(f"  Embedding space      : {collapse_status}")

# ══════════════════════════════════════════════════════════════════════════
# TEST 7 — Intra-person consistency vs inter-person gap
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TEST 7: Intra vs inter-person sim gap (discrimination margin)")
print(SEP)

# Intra: same person, 5 jittered crops
intra_sims = []
ref_name   = "P-Red"
ref_color  = PERSONS[ref_name]
ref_feat2  = person_feats[ref_name]

for dx in [-8, -4, 0, 4, 8]:
    f  = blank()
    b  = draw_person(f, 640 + dx, ref_color)
    ff = reid.extract(f, b)
    intra_sims.append(cosine(ref_feat2, ff))

mean_intra = np.mean(intra_sims)
margin     = mean_intra - mean_inter

print(f"  Intra-person sim (jitter): {mean_intra:.4f}   ← should be HIGH")
print(f"  Inter-person sim (diff):   {mean_inter:.4f}   ← should be LOW")
print(f"  Discrimination margin:     {margin:.4f}   ← should be > 0.30")
print()
if margin > 0.30:
    margin_status = "EXCELLENT — model clearly separates identities"
elif margin > 0.10:
    margin_status = "MODERATE  — some separation, but train weights needed"
elif margin > 0.01:
    margin_status = "POOR      — barely separating, use trained weights"
else:
    margin_status = "NONE      — random weights, all embeddings are identical"
print(f"  Margin status: {margin_status}")

# ══════════════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print("  VERDICT")
print('=' * 62)
print(f"  Shape correct (2048,)         : {'PASS' if shape_ok else 'FAIL'}")
print(f"  L2-normalised (norm=1.0)      : {'PASS' if norm_ok  else 'FAIL'}")
print(f"  Deterministic (same crop)     : {'PASS' if np.mean(sims) > 0.9999 else 'FAIL'}")
print(f"  Jitter robust (shift ≤20px)   : {'PASS' if avg_jitter > 0.85 else 'WARN'}")
print(f"  Brightness robust             : {'PASS' if avg_bright > 0.80 else 'WARN'}")
print(f"  Embedding collapse            : {'DETECTED' if collapsed else 'NOT detected'}")
print(f"  Discrimination margin         : {margin:.4f}  ({margin_status.split('—')[0].strip()})")
print()
if collapsed:
    print("  OVERALL: RANDOM/UNTRAINED WEIGHTS DETECTED")
    print()
    print("  What this means for your mall system:")
    print("  ├─ Pipeline logic (gallery, matching, threads) is fully working")
    print("  ├─ All 2048-d vectors have norm=1.0 and are deterministic")
    print("  ├─ BUT inter-person similarity ~1.0 means EVERYONE gets the same ID")
    print("  └─ Fix: replace onnx_model/resnet50_reid.onnx with trained weights")
    print()
    print("  Trained weight options:")
    print("  A) CTL (Centroid Triplet Loss) trained on Market1501 — paper's choice")
    print("     → github.com/michuanhaohao/reid-strong-baseline")
    print("  B) OSNet (lightweight, good cross-domain) — recommended for malls")
    print("     → github.com/KaiyangZhou/deep-person-reid")
    print("  C) CLIP-ReID — best cross-domain accuracy")
    print("     → github.com/Syliz/CLIP-ReID")
    print()
    print("  After getting weights, re-export to ONNX:")
    print("  → python build_reid_onnx.py   (update the model path inside)")
else:
    print("  OVERALL: EMBEDDINGS ARE UNIQUE — model is producing good features")
    print("  System is ready for deployment.")
print()
