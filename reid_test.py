"""
reid_test.py — Per-Camera ReID Tracker
Pipeline: YOLOv8m → ResNet50 ReID ONNX → IoU tracking + ReID on exit/entry

Cross-camera identity:
  - Each camera runs its own PersonTracker
  - All trackers share a single GlobalReIDGallery (injected by camera_manager.py)
  - When a person exits any camera their features are stored in the global gallery
  - When they appear in ANY other camera the same Global ID (G1, G2 …) is reused
  - Within a single camera IoU spatial matching is used first

Retention policy (v2 — mall/public-space edition):
  - All expiry is WALL-CLOCK based (seconds), not frame-count based
  - Default retention: 12 hours (43200 seconds)
  - wall_time resets every time a person is seen again → active visitors never expire
  - Only someone genuinely absent for 12+ consecutive hours is purged
"""

import cv2
import numpy as np
import onnxruntime as ort
import threading
import time

# =============================================================================
# !! EDIT THESE — shared defaults (camera_manager.py can override per-camera) !!
# =============================================================================

YOLO_MODEL  = "yolov8m.pt"
REID_MODEL  = "onnx_model/resnet50_reid.onnx"

CONF_THRESH              = 0.50    # YOLO min confidence
IOU_THRESH               = 0.20    # IoU threshold for within-camera spatial match
REID_THRESH              = 0.70    # Cosine similarity — within-camera re-entry
CROSS_CAM_THRESH         = 0.72    # Cosine similarity — cross-camera match
MAX_LOST_FRAMES          = 60      # Frames before retiring a track to the gallery
REENTRY_WINDOW_SECONDS   = 43200.0 # 12 hours — wall-clock retention for mall/public spaces

# =============================================================================
# Colour palette (cycles by Global ID number)
# =============================================================================

COLORS = [
    (255, 80,  80),  (80,  255, 80),  (80,  80,  255),
    (255, 200, 50),  (200, 50,  255), (50,  200, 255),
    (255, 120, 200), (120, 255, 150), (150, 120, 255),
    (255, 180, 100), (100, 255, 200), (180, 100, 255),
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =============================================================================
# GlobalReIDGallery  — ONE instance shared across ALL cameras
# =============================================================================

class GlobalReIDGallery:
    """
    Thread-safe gallery that:
      • issues globally unique IDs  (G1, G2, G3 …)
      • stores feature snapshots when a person exits any camera
      • matches those features when a person appears in any camera
      • retains identities for up to reentry_window_seconds of INACTIVITY
        (wall_time resets on every store() or find_match() hit, so active
        visitors who keep appearing will never be purged during the day)

    Designed for shopping malls / public spaces:
      Default retention = 12 hours (43200 seconds).
      A shopper who leaves for lunch and returns 3 hours later keeps their ID.
      A shopper last seen 13 hours ago is quietly removed to free memory.
    """

    def __init__(self,
                 cross_cam_thresh: float        = CROSS_CAM_THRESH,
                 reentry_window_seconds: float  = REENTRY_WINDOW_SECONDS):

        self._lock                  = threading.Lock()
        self._next_id               = 1
        self._stored: dict          = {}   # gid → {"features", "wall_time", "last_cam", "first_seen"}
        self.cross_cam_thresh       = cross_cam_thresh
        self.reentry_window_seconds = reentry_window_seconds

    # ── ID issuing ─────────────────────────────────────────────────────────

    def issue_id(self) -> str:
        with self._lock:
            gid = f"G{self._next_id}"
            self._next_id += 1
            return gid

    @property
    def total_issued(self) -> int:
        with self._lock:
            return self._next_id - 1

    # ── Feature storage (called when a person exits a camera) ──────────────

    def store(self, gid: str, features: list, cam_id: int):
        """
        Save/refresh feature snapshots for gid.
        Keeps up to 8 most recent snapshots.
        Resets wall_time so the 12-hr inactivity clock restarts.
        """
        with self._lock:
            entry = self._stored.get(gid)
            if entry is None:
                self._stored[gid] = {
                    "features":   list(features[-8:]),
                    "wall_time":  time.time(),
                    "first_seen": time.time(),
                    "last_cam":   cam_id,
                }
            else:
                merged = entry["features"] + list(features)
                entry["features"]  = merged[-8:]
                entry["wall_time"] = time.time()   # resets inactivity clock
                entry["last_cam"]  = cam_id

    # ── Matching (called when a new detection cannot match active tracks) ──

    def find_match(self, det_feat: np.ndarray, requesting_cam_id: int):
        """
        Search the gallery for a person matching det_feat.
        Returns (gid, similarity) or (None, 0.0).
        On a successful match, wall_time is refreshed so the inactivity
        clock resets — the person won't expire while still in the building.
        """
        self._purge_old()

        best_gid   = None
        best_score = 0.0

        with self._lock:
            for gid, info in self._stored.items():
                if info["last_cam"] == requesting_cam_id:
                    continue
                sim = max(
                    float(np.dot(det_feat, f) /
                          (np.linalg.norm(det_feat) * np.linalg.norm(f) + 1e-6))
                    for f in info["features"]
                )
                if sim >= self.cross_cam_thresh and sim > best_score:
                    best_score = sim
                    best_gid   = gid

            # Refresh wall_time on match — active visitors never expire
            if best_gid is not None:
                self._stored[best_gid]["wall_time"] = time.time()

        return best_gid, best_score

    def _purge_old(self):
        """
        Remove entries inactive longer than reentry_window_seconds.
        Because wall_time resets on every store() and find_match() hit,
        only people genuinely absent for 12+ hours are removed.
        """
        now = time.time()
        with self._lock:
            old = [
                gid for gid, info in self._stored.items()
                if now - info["wall_time"] > self.reentry_window_seconds
            ]
            for gid in old:
                del self._stored[gid]
            if old:
                print(f"[Gallery] Purged {len(old)} entries inactive > "
                      f"{self.reentry_window_seconds/3600:.1f}h: "
                      f"{old[:5]}{'...' if len(old) > 5 else ''}")

    def gallery_size(self) -> int:
        with self._lock:
            return len(self._stored)

    def summary(self) -> dict:
        """Return a snapshot of gallery stats (thread-safe)."""
        now = time.time()
        with self._lock:
            sizes = [len(v["features"]) for v in self._stored.values()]
            ages  = [(now - v["first_seen"]) / 3600 for v in self._stored.values()]
            return {
                "total_ids":    self.total_issued,
                "gallery_size": len(self._stored),
                "avg_snapshots": round(sum(sizes) / len(sizes), 1) if sizes else 0,
                "oldest_hrs":   round(max(ages), 2) if ages else 0,
                "newest_hrs":   round(min(ages), 2) if ages else 0,
            }


# =============================================================================
# ResNet50 ReID Feature Extractor
# =============================================================================

class ReIDExtractor:
    def __init__(self, model_path: str = REID_MODEL):
        print(f"[ReID] Loading: {model_path}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session     = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[ReID] Provider: {self.session.get_providers()[0]}")

    def preprocess(self, frame: np.ndarray, bbox):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1));  y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w)); y2 = max(y1 + 1, min(y2, h))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (128, 256))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - MEAN) / STD
        return crop.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def extract(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        inp = self.preprocess(frame, bbox)
        if inp is None:
            return None
        feat = self.session.run([self.output_name], {self.input_name: inp})[0].flatten()
        norm = np.linalg.norm(feat)
        return (feat / norm).astype(np.float32) if norm > 1e-6 else None


# =============================================================================
# Per-camera track object
# =============================================================================

class Track:
    def __init__(self, gid: str, bbox, features: np.ndarray, frame_id: int):
        self.id           = gid
        self.bbox         = bbox
        self.features     = features
        self.all_features = [features]
        self.lost         = 0
        self.frame_id     = frame_id
        self.confirmed    = False
        self.hits         = 1

    def update(self, bbox, features: np.ndarray):
        self.bbox     = bbox
        self.features = features
        self.all_features.append(features)
        if len(self.all_features) > 10:
            self.all_features.pop(0)
        self.lost  = 0
        self.hits += 1
        if self.hits >= 2:
            self.confirmed = True

    def best_sim(self, features: np.ndarray) -> float:
        return max(
            float(np.dot(features, f) /
                  (np.linalg.norm(features) * np.linalg.norm(f) + 1e-6))
            for f in self.all_features
        )


# =============================================================================
# Per-camera PersonTracker
# =============================================================================

def _iou(b1, b2) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """
    One instance per camera.
    Inject a shared GlobalReIDGallery so cross-camera identity works.
    local_exited uses wall-clock time (not frame count).
    """

    def __init__(self,
                 reid:                   ReIDExtractor,
                 global_gallery:         GlobalReIDGallery,
                 cam_id:                 int,
                 iou_thresh:             float = IOU_THRESH,
                 reid_thresh:            float = REID_THRESH,
                 max_lost_frames:        int   = MAX_LOST_FRAMES,
                 reentry_window_seconds: float = REENTRY_WINDOW_SECONDS):

        self.reid                   = reid
        self.gallery                = global_gallery
        self.cam_id                 = cam_id
        self.iou_thresh             = iou_thresh
        self.reid_thresh            = reid_thresh
        self.max_lost               = max_lost_frames
        self.reentry_window_seconds = reentry_window_seconds

        self.tracks:       dict = {}   # gid → Track  (active in this camera)
        self.local_exited: dict = {}   # gid → {"features", "wall_time"}
        self.frame_id           = 0

    # ── Internal helpers ───────────────────────────────────────────────────

    def _match_active(self, bbox, feat):
        best_gid   = None
        best_score = -1.0
        for gid, track in self.tracks.items():
            if track.lost > 5:
                continue
            spatial = _iou(bbox, track.bbox)
            appear  = track.best_sim(feat)
            if spatial >= self.iou_thresh:
                score = 0.6 * spatial + 0.4 * appear
            elif appear >= self.reid_thresh:
                score = 0.3 * spatial + 0.7 * appear
            else:
                continue
            if score > best_score:
                best_score = score
                best_gid   = gid
        return best_gid, best_score

    def _match_local_exited(self, feat):
        """Same-camera re-entry — wall-clock based expiry."""
        best_gid   = None
        best_score = 0.0
        now        = time.time()
        for gid, info in self.local_exited.items():
            if now - info["wall_time"] > self.reentry_window_seconds:
                continue
            sim = max(
                float(np.dot(feat, f) /
                      (np.linalg.norm(feat) * np.linalg.norm(f) + 1e-6))
                for f in info["features"]
            )
            if sim >= self.reid_thresh and sim > best_score:
                best_score = sim
                best_gid   = gid
        return best_gid, best_score

    # ── Main update ────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, detections: list) -> list:
        """
        detections: list of [x1, y1, x2, y2, conf]
        Returns:    list of (gid, bbox, conf, is_new, is_reentry, is_cross_cam)
        """
        self.frame_id += 1

        for track in self.tracks.values():
            track.lost += 1

        results      = []
        matched_gids = set()

        det_data = []
        for det in detections:
            bbox = det[:4]
            feat = self.reid.extract(frame, bbox)
            det_data.append((bbox, det[4], feat))

        # ── Pass 1: match active tracks (IoU + appearance) ─────────────────
        unmatched = []
        for bbox, conf, feat in det_data:
            if feat is None:
                continue
            gid, _ = self._match_active(bbox, feat)
            if gid is not None and gid not in matched_gids:
                self.tracks[gid].update(bbox, feat)
                matched_gids.add(gid)
                results.append((gid, bbox, conf, False, False, False))
            else:
                unmatched.append((bbox, conf, feat))

        # ── Pass 2: local exit gallery (same camera re-entry) ───────────────
        still_unmatched = []
        for bbox, conf, feat in unmatched:
            gid, score = self._match_local_exited(feat)
            if gid is not None:
                self.tracks[gid] = Track(gid, bbox, feat, self.frame_id)
                self.tracks[gid].confirmed = True
                del self.local_exited[gid]
                matched_gids.add(gid)
                results.append((gid, bbox, conf, False, True, False))
                print(f"  [Cam {self.cam_id}] ↩  Local re-entry {gid} (sim={score:.3f})")
            else:
                still_unmatched.append((bbox, conf, feat))

        # ── Pass 3: global gallery (cross-camera re-identification) ─────────
        truly_new = []
        for bbox, conf, feat in still_unmatched:
            gid, score = self.gallery.find_match(feat, self.cam_id)
            if gid is not None:
                self.tracks[gid] = Track(gid, bbox, feat, self.frame_id)
                self.tracks[gid].confirmed = True
                matched_gids.add(gid)
                results.append((gid, bbox, conf, False, False, True))
                print(f"  [Cam {self.cam_id}] 🔁 Cross-cam match {gid} (sim={score:.3f})")
            else:
                truly_new.append((bbox, conf, feat))

        # ── Pass 4: genuinely new people ────────────────────────────────────
        for bbox, conf, feat in truly_new:
            gid = self.gallery.issue_id()
            self.tracks[gid] = Track(gid, bbox, feat, self.frame_id)
            matched_gids.add(gid)
            results.append((gid, bbox, conf, True, False, False))

        # ── Retire lost tracks ───────────────────────────────────────────────
        to_delete = []
        for gid, track in self.tracks.items():
            if track.lost > self.max_lost:
                if track.confirmed:
                    self.local_exited[gid] = {
                        "features":  track.all_features[-5:],
                        "wall_time": time.time(),
                    }
                    self.gallery.store(gid, track.all_features[-5:], self.cam_id)
                    print(f"  [Cam {self.cam_id}] 📤 Stored {gid} in global gallery")
                to_delete.append(gid)
        for gid in to_delete:
            del self.tracks[gid]

        # Purge stale local exits — wall-clock based
        now = time.time()
        old = [g for g, info in self.local_exited.items()
               if now - info["wall_time"] > self.reentry_window_seconds]
        for g in old:
            del self.local_exited[g]

        return results


# =============================================================================
# Drawing helper
# =============================================================================

def draw_frame(frame: np.ndarray,
               results: list,
               tracker: PersonTracker,
               fps: float,
               cam_label: str = "") -> np.ndarray:
    """
    Annotate a single camera frame with bounding boxes, IDs and stats.
    results: list of (gid, bbox, conf, is_new, is_reentry, is_cross_cam)
    """
    for gid, bbox, conf, is_new, is_reentry, is_cross_cam in results:
        x1, y1, x2, y2 = [int(c) for c in bbox]
        try:
            num = int(gid.lstrip("G"))
        except ValueError:
            num = 0
        color     = COLORS[(num - 1) % len(COLORS)]
        track     = tracker.tracks.get(gid)
        confirmed = track.confirmed if track else True

        thickness = 2 if confirmed else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if is_new:
            tag = "NEW"
        elif is_cross_cam:
            tag = "XCAM"
        elif is_reentry:
            tag = "RETURN"
        else:
            tag = ""

        label = f"{gid}  {conf:.2f}  {tag}".strip()
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    active  = sum(1 for t in tracker.tracks.values() if t.confirmed)
    gallery = tracker.gallery.gallery_size()
    stats = [
        cam_label,
        f"FPS: {fps:.1f}",
        f"Visible: {len(results)}",
        f"Active: {active}",
        f"Global gallery: {gallery}",
        f"Total IDs: {tracker.gallery.total_issued}",
    ]
    y = 22
    for s in stats:
        if s:
            cv2.putText(frame, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)
            y += 22

    return frame
