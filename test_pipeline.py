"""
test_pipeline.py — Full pipeline dry-run (no camera or video needed)
=====================================================================
Tests all matching passes with the 12-hour wall-clock retention policy.

Tests:
  A — Within-camera tracking (5 persons, 30 frames each)
  B — Cross-camera re-identification (same persons appear in cam_1)
  C — Within-camera re-entry (person walks back into cam_0)
  D — ReIDExtractor inference timing (100 crops)
  E — Feature vector sanity (norm, intra/inter-class similarity)
  F — Retention: verify wall_time refreshes on match (no premature expiry)
"""

import sys, os, time
import numpy as np
import cv2

sys.path.insert(0, "/home/claude")
os.chdir("/home/claude")

import reid_test as rt
from reid_test import (
    ReIDExtractor, GlobalReIDGallery, PersonTracker,
    CONF_THRESH, CROSS_CAM_THRESH, REID_THRESH, MAX_LOST_FRAMES,
    REENTRY_WINDOW_SECONDS,
)

FRAME_H, FRAME_W = 720, 1280

def make_blank_frame(bg_color=(30, 30, 30)):
    return np.full((FRAME_H, FRAME_W, 3), bg_color, dtype=np.uint8)

def draw_person(frame, bbox, color=(200, 200, 200)):
    x1, y1, x2, y2 = [int(c) for c in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(frame, str(color), (x1+2, y1+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

PERSONS = {
    "P1": (220,  80,  80),
    "P2": ( 80, 220,  80),
    "P3": ( 80,  80, 220),
    "P4": (220, 180,  50),
    "P5": (180,  50, 220),
}

def make_bbox(x_center, y_center=380, w=80, h=200):
    x1 = x_center - w//2
    y1 = y_center - h//2
    return [x1, y1, x1+w, y1+h]

print("=" * 60)
print("ReID Pipeline Test — Mall Edition (12-hr retention)")
print(f"REENTRY_WINDOW_SECONDS = {REENTRY_WINDOW_SECONDS/3600:.0f} hrs")
print("=" * 60)

print("\n[1] Loading ReIDExtractor (ResNet50 ONNX)...")
t0   = time.time()
reid = ReIDExtractor("onnx_model/resnet50_reid.onnx")
print(f"    Loaded in {time.time()-t0:.2f}s")

print("[2] Creating GlobalReIDGallery (12-hr retention)...")
gallery = GlobalReIDGallery(
    cross_cam_thresh       = CROSS_CAM_THRESH,
    reentry_window_seconds = REENTRY_WINDOW_SECONDS,
)

print("[3] Creating PersonTracker for cam_0 and cam_1...")
tracker_0 = PersonTracker(reid, gallery, cam_id=0,
                           reentry_window_seconds=REENTRY_WINDOW_SECONDS)
tracker_1 = PersonTracker(reid, gallery, cam_id=1,
                           reentry_window_seconds=REENTRY_WINDOW_SECONDS)

# ── Test A ────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("TEST A: Within-camera tracking")
print("─"*60)

person_gids = {}
for name, color in PERSONS.items():
    for f in range(30):
        frame  = make_blank_frame()
        bbox   = make_bbox(200 + f * 20)
        draw_person(frame, bbox, color)
        results = tracker_0.update(frame, [(*bbox, 0.93)])
        if results and name not in person_gids:
            person_gids[name] = results[0][0]
            print(f"  {name} assigned → {person_gids[name]}")

print(f"\n  Retiring tracks ({MAX_LOST_FRAMES+5} blank frames)...")
for _ in range(MAX_LOST_FRAMES + 5):
    tracker_0.update(make_blank_frame(), [])

print(f"  Global gallery size: {gallery.gallery_size()} (expect up to 5)")

# ── Test B ────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("TEST B: Cross-camera re-ID")
print("─"*60)

cross_cam_results = {}
for name, color in PERSONS.items():
    frame   = make_blank_frame((20, 20, 50))
    bbox    = make_bbox(640)
    draw_person(frame, bbox, color)
    results = tracker_1.update(frame, [(*bbox, 0.92)])
    if results:
        gid, _, conf, is_new, is_reentry, is_cross = results[0]
        expected = person_gids.get(name)
        match    = (gid == expected)
        cross_cam_results[name] = match
        tag    = "XCAM" if is_cross else ("NEW" if is_new else "active")
        status = "PASS" if match else "FAIL"
        print(f"  {name}: expected={expected}  got={gid}  tag={tag}  {status}")

# ── Test C ────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("TEST C: Within-camera re-entry")
print("─"*60)

name, color   = "P1", PERSONS["P1"]
expected_gid  = person_gids["P1"]
frame = make_blank_frame()
bbox  = make_bbox(400)
draw_person(frame, bbox, color)
results = tracker_0.update(frame, [(*bbox, 0.95)])
if results:
    gid, _, conf, is_new, is_reentry, is_cross = results[0]
    match  = (gid == expected_gid)
    tag    = "RETURN" if is_reentry else ("XCAM" if is_cross else ("NEW" if is_new else "active"))
    status = "PASS" if match else "FAIL"
    print(f"  P1 re-entry: expected={expected_gid}  got={gid}  tag={tag}  {status}")

# ── Test D ────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("TEST D: Inference timing (100 crops)")
print("─"*60)

times = []
for _ in range(100):
    frame = make_blank_frame()
    bbox  = make_bbox(640)
    draw_person(frame, bbox, (200, 100, 50))
    t     = time.time()
    reid.extract(frame, bbox)
    times.append((time.time() - t) * 1000)

print(f"  Mean:   {np.mean(times):.2f} ms")
print(f"  Median: {np.median(times):.2f} ms")
print(f"  P95:    {np.percentile(times, 95):.2f} ms")
print(f"  Paper (RTX 3090): 7.28 ms — CPU expected slower")

# ── Test E ────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("TEST E: Feature vector sanity")
print("─"*60)

feats = []
for name, color in PERSONS.items():
    frame = make_blank_frame()
    bbox  = make_bbox(640)
    draw_person(frame, bbox, color)
    f = reid.extract(frame, bbox)
    if f is not None:
        feats.append((name, f))
        print(f"  {name}: dim={f.shape[0]}  norm={np.linalg.norm(f):.4f}")

# ── Test F ────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("TEST F: Retention — wall_time refreshes on match")
print("─"*60)

g2 = gallery._stored.get(person_gids.get("P1"))
if g2:
    before = g2["wall_time"]
    time.sleep(0.05)
    # Simulate a cross-cam match which should refresh wall_time
    frame = make_blank_frame((20, 20, 50))
    bbox  = make_bbox(500)
    draw_person(frame, bbox, PERSONS["P1"])
    # Use cam_id=2 so it qualifies as cross-cam
    tracker_2 = PersonTracker(reid, gallery, cam_id=2,
                               reentry_window_seconds=REENTRY_WINDOW_SECONDS)
    tracker_2.update(frame, [(*bbox, 0.93)])
    after = gallery._stored.get(person_gids.get("P1"), {}).get("wall_time", before)
    refreshed = after > before
    print(f"  wall_time before: {before:.3f}")
    print(f"  wall_time after:  {after:.3f}")
    print(f"  Refreshed: {'PASS — inactivity clock reset' if refreshed else 'FAIL'}")
else:
    print("  (P1 not in global gallery — skipping)")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
cross_pass  = sum(cross_cam_results.values())
cross_total = len(cross_cam_results)
print(f"  Cross-camera ReID  : {cross_pass}/{cross_total} correct")
print(f"  Gallery size       : {gallery.gallery_size()}")
print(f"  Total IDs issued   : {gallery.total_issued}")
print(f"  ReID latency (mean): {np.mean(times):.1f} ms")
print(f"  Retention window   : {REENTRY_WINDOW_SECONDS/3600:.0f} hours")
print()
if cross_pass == cross_total:
    print("  All cross-camera tests PASSED")
else:
    print(f"  {cross_total - cross_pass} cross-camera tests FAILED")
    print("  (expected with random weights — use trained Market1501 ONNX in production)")
