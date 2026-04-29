"""
camera_manager.py — Multi-Camera ReID Orchestrator
====================================================
Manages up to N Android phones streaming via RTSP (IP Webcam app).
Each phone = one zone/camera.  A single GlobalReIDGallery ensures that a
person walking from Zone 1 → Zone 2 keeps the same Global ID (G1, G2 …).

Retention policy (v2 — mall/public-space edition):
  - IDs retained for 12 hours of INACTIVITY (wall-clock based)
  - A shopper last seen 11 hours ago still matches when they return
  - wall_time resets on every sighting so active visitors never expire
  - Gallery status logged every 5 minutes to terminal

Architecture
------------
  camera_manager.py
  ├── GlobalReIDGallery          (shared, thread-safe, 12-hr retention)
  ├── GalleryStatusLogger        (daemon thread — logs every 5 min)
  ├── CameraWorker × N           (one thread per RTSP source)
  │     ├── cv2.VideoCapture     (RTSP stream, auto-reconnects)
  │     ├── YOLO (shared model)  (person detection)
  │     ├── ReIDExtractor        (shared ONNX session)
  │     └── PersonTracker        (per-camera, 12-hr local_exited)
  └── DisplayManager             (tiled preview window)

Quick-start
-----------
1. Install IP Webcam on each Android phone and tap "Start server".
2. Note the IP shown on each phone screen.
3. Fill in CAMERA_SOURCES below with each phone's RTSP URL.
4. Run:  python camera_manager.py

RTSP URL format for IP Webcam:
    rtsp://<phone-ip>:8080/h264_ulaw.sdp

Press Q in the preview window to stop all cameras.
Press S in the preview window to print a live summary to the terminal.
"""

import cv2
import numpy as np
import threading
import time
import queue
import datetime
from ultralytics import YOLO

from reid_test import (
    ReIDExtractor,
    GlobalReIDGallery,
    PersonTracker,
    draw_frame,
    YOLO_MODEL,
    REID_MODEL,
    CONF_THRESH,
    IOU_THRESH,
    REID_THRESH,
    CROSS_CAM_THRESH,
    MAX_LOST_FRAMES,
    REENTRY_WINDOW_SECONDS,
)

# =============================================================================
# !! EDIT THESE !!
# =============================================================================

# Each entry: (label, rtsp_url_or_int)
# Use 0,1,2… for local webcams while testing without phones.
# Replace with real RTSP URLs once phones are running IP Webcam.
CAMERA_SOURCES = [
    ("Zone-1", "rtsp://192.168.1.101:8080/h264_ulaw.sdp"),
    ("Zone-2", "rtsp://192.168.1.102:8080/h264_ulaw.sdp"),
    ("Zone-3", "rtsp://192.168.1.103:8080/h264_ulaw.sdp"),
    ("Zone-4", "rtsp://192.168.1.104:8080/h264_ulaw.sdp"),
    ("Zone-5", "rtsp://192.168.1.105:8080/h264_ulaw.sdp"),
]

SAVE_OUTPUTS   = False    # True → save annotated .mp4 per camera
DISPLAY_WIDTH  = 640      # Width of each tile in the preview grid
DISPLAY_HEIGHT = 360      # Height of each tile

# Grid layout: 3 cols × 2 rows fits 5 cameras (last cell blank)
GRID_COLS      = 3
GRID_ROWS      = 2

# Reconnect after this many seconds if a stream drops
RECONNECT_DELAY = 5.0

# How often (seconds) to print gallery status to terminal
STATUS_LOG_INTERVAL = 300   # 5 minutes

# Retention: how long (seconds) to remember a person after last sighting
# 43200 = 12 hours | 86400 = 24 hours
RETENTION_SECONDS = REENTRY_WINDOW_SECONDS   # pulled from reid_test.py (43200)


# =============================================================================
# GalleryStatusLogger — daemon thread, logs every STATUS_LOG_INTERVAL seconds
# =============================================================================

class GalleryStatusLogger(threading.Thread):
    """Periodically prints gallery stats so you can monitor the system."""

    def __init__(self, gallery: GlobalReIDGallery,
                 workers_ref: list,
                 interval: float = STATUS_LOG_INTERVAL):
        super().__init__(daemon=True, name="GalleryLogger")
        self.gallery  = gallery
        self.workers  = workers_ref
        self.interval = interval

    def run(self):
        while True:
            time.sleep(self.interval)
            now = datetime.datetime.now().strftime("%H:%M:%S")
            s   = self.gallery.summary()
            print(f"\n[{now}] ── Gallery status ─────────────────────────────")
            print(f"  Unique IDs issued  : {s['total_ids']}")
            print(f"  Currently in store : {s['gallery_size']}")
            print(f"  Avg feature snaps  : {s['avg_snapshots']}")
            print(f"  Oldest entry       : {s['oldest_hrs']} hrs ago")
            print(f"  Newest entry       : {s['newest_hrs']} hrs ago")
            for w in self.workers:
                active = sum(1 for t in w.tracker.tracks.values() if t.confirmed)
                print(f"  [{w.label:8s}] active={active}  "
                      f"local_exit={len(w.tracker.local_exited)}  "
                      f"fps={w.fps_display:.1f}  "
                      f"{'Live' if w.connected else 'OFFLINE'}")
            print("──────────────────────────────────────────────────────\n")


# =============================================================================
# CameraWorker — runs in its own thread
# =============================================================================

class CameraWorker(threading.Thread):
    """
    Captures frames from one RTSP source, runs YOLO + ReID tracking,
    and puts annotated frames into a shared output queue.
    Auto-reconnects on stream drop.
    """

    def __init__(self,
                 cam_id:    int,
                 label:     str,
                 source,                      # str (RTSP URL) or int (webcam index)
                 yolo:      YOLO,
                 reid:      ReIDExtractor,
                 gallery:   GlobalReIDGallery,
                 out_queue: queue.Queue,
                 save:      bool = False):

        super().__init__(daemon=True, name=f"CamWorker-{cam_id}")
        self.cam_id    = cam_id
        self.label     = label
        self.source    = source
        self.yolo      = yolo
        self.reid      = reid
        self.gallery   = gallery
        self.out_queue = out_queue
        self.save      = save

        self.tracker = PersonTracker(
            reid                   = reid,
            global_gallery         = gallery,
            cam_id                 = cam_id,
            iou_thresh             = IOU_THRESH,
            reid_thresh            = REID_THRESH,
            max_lost_frames        = MAX_LOST_FRAMES,
            reentry_window_seconds = RETENTION_SECONDS,
        )

        self._stop_event = threading.Event()
        self._writer     = None
        self.fps_display = 0.0
        self.connected   = False

    def stop(self):
        self._stop_event.set()

    def _open_source(self) -> cv2.VideoCapture:
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        # Keep buffer at 1 frame — prevents stale-frame lag on RTSP
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            print(f"[Cam {self.cam_id} | {self.label}] Connected → {self.source}")
            self.connected = True
        else:
            print(f"[Cam {self.cam_id} | {self.label}] Cannot open: {self.source}")
            self.connected = False
        return cap

    def _init_writer(self, w: int, h: int, fps: float):
        if self.save:
            fname  = f"output_cam{self.cam_id}_{self.label.replace(' ', '_')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(fname, fourcc, fps, (w, h))
            print(f"[Cam {self.cam_id}] Saving to {fname}")

    def run(self):
        while not self._stop_event.is_set():
            cap = self._open_source()
            if not cap.isOpened():
                blank = _blank_tile(self.label, "NO SIGNAL", DISPLAY_WIDTH, DISPLAY_HEIGHT)
                self.out_queue.put((self.cam_id, blank))
                time.sleep(RECONNECT_DELAY)
                continue

            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._init_writer(w, h, fps)

            fps_count = 0
            fps_start = time.time()

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"[Cam {self.cam_id}] Stream lost — reconnecting in {RECONNECT_DELAY}s …")
                    self.connected = False
                    break

                # ── YOLO detection (person class = 0) ───────────────────────
                yolo_out   = self.yolo(frame, classes=[0],
                                       conf=CONF_THRESH, verbose=False)[0]
                detections = []
                if yolo_out.boxes is not None:
                    for box in yolo_out.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, float(box.conf[0])])

                # ── ReID tracking ────────────────────────────────────────────
                results = self.tracker.update(frame, detections)

                # ── FPS counter ──────────────────────────────────────────────
                fps_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    self.fps_display = fps_count / elapsed
                    fps_count = 0
                    fps_start = time.time()

                # ── Annotate frame ───────────────────────────────────────────
                annotated = draw_frame(
                    frame.copy(), results, self.tracker,
                    self.fps_display,
                    cam_label=f"[{self.label}] Cam {self.cam_id}",
                )

                if self._writer:
                    self._writer.write(annotated)

                # ── Resize for display grid ──────────────────────────────────
                tile = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                try:
                    self.out_queue.put_nowait((self.cam_id, tile))
                except queue.Full:
                    pass

            cap.release()
            if self._writer:
                self._writer.release()
                self._writer = None

            if not self._stop_event.is_set():
                time.sleep(RECONNECT_DELAY)

        print(f"[Cam {self.cam_id}] Worker stopped.")


# =============================================================================
# DisplayManager — assembles tile grid and shows it
# =============================================================================

class DisplayManager:
    def __init__(self, n_cams: int, cols: int, rows: int,
                 tile_w: int, tile_h: int):
        self.n_cams = n_cams
        self.cols   = cols
        self.rows   = rows
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.tiles  = {
            i: _blank_tile(f"Zone-{i+1}", "Waiting…", tile_w, tile_h)
            for i in range(n_cams)
        }

    def update(self, cam_id: int, tile: np.ndarray):
        self.tiles[cam_id] = tile

    def render(self) -> np.ndarray:
        blank     = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.uint8)
        rows_imgs = []
        for r in range(self.rows):
            row_tiles = []
            for c in range(self.cols):
                idx  = r * self.cols + c
                tile = self.tiles.get(idx, blank) if idx < self.n_cams else blank
                row_tiles.append(tile)
            rows_imgs.append(np.hstack(row_tiles))
        return np.vstack(rows_imgs)


# =============================================================================
# Utility helpers
# =============================================================================

def _blank_tile(label: str, msg: str, w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 100), 2)
    cv2.putText(img, msg,   (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,  80,  200), 2)
    return img


def print_summary(workers: list, gallery: GlobalReIDGallery):
    s   = gallery.summary()
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n[{now}] ── Live summary ──────────────────────────────────")
    print(f"  Unique IDs issued  : {s['total_ids']}")
    print(f"  Gallery size       : {s['gallery_size']}")
    print(f"  Avg feature snaps  : {s['avg_snapshots']}")
    print(f"  Oldest entry       : {s['oldest_hrs']} hrs")
    print(f"  Newest entry       : {s['newest_hrs']} hrs")
    for w in workers:
        active = sum(1 for t in w.tracker.tracks.values() if t.confirmed)
        print(f"  [{w.label:8s}] active={active:3d}  "
              f"local_exit={len(w.tracker.local_exited):3d}  "
              f"fps={w.fps_display:.1f}  "
              f"{'Live' if w.connected else 'OFFLINE'}")
    print("──────────────────────────────────────────────────────\n")


# =============================================================================
# Main
# =============================================================================

def main():
    n_cams = len(CAMERA_SOURCES)
    retention_hrs = RETENTION_SECONDS / 3600

    print("\n" + "=" * 60)
    print("  Multi-Camera ReID Manager — Mall / Public Space Edition")
    print(f"  Cameras      : {n_cams}")
    print(f"  YOLO model   : {YOLO_MODEL}")
    print(f"  ReID model   : {REID_MODEL}")
    print(f"  Cross-cam ≥  : {CROSS_CAM_THRESH}")
    print(f"  ID retention : {retention_hrs:.0f} hours (inactivity-based)")
    print(f"  Status log   : every {STATUS_LOG_INTERVAL//60} min")
    print(f"  Save outputs : {SAVE_OUTPUTS}")
    print("=" * 60 + "\n")

    # ── Load models once — shared across all camera threads ──────────────
    print("[YOLO] Loading …")
    yolo = YOLO(YOLO_MODEL)
    print("[YOLO] Ready.\n")

    reid = ReIDExtractor(REID_MODEL)

    gallery = GlobalReIDGallery(
        cross_cam_thresh       = CROSS_CAM_THRESH,
        reentry_window_seconds = RETENTION_SECONDS,
    )

    # ── Shared frame queue ────────────────────────────────────────────────
    frame_queue = queue.Queue(maxsize=2 * n_cams)

    # ── Start one worker per camera ───────────────────────────────────────
    workers = []
    for cam_id, (label, source) in enumerate(CAMERA_SOURCES):
        worker = CameraWorker(
            cam_id    = cam_id,
            label     = label,
            source    = source,
            yolo      = yolo,
            reid      = reid,
            gallery   = gallery,
            out_queue = frame_queue,
            save      = SAVE_OUTPUTS,
        )
        workers.append(worker)
        worker.start()
        print(f"[Manager] Started worker for {label} ({source})")

    # ── Start gallery status logger ───────────────────────────────────────
    logger = GalleryStatusLogger(gallery, workers, STATUS_LOG_INTERVAL)
    logger.start()

    print(f"\n[Manager] All {n_cams} workers started.")
    print("  Press  Q  in the preview window to quit")
    print("  Press  S  in the preview window to print a live summary\n")

    display = DisplayManager(n_cams, GRID_COLS, GRID_ROWS,
                             DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # ── Main display loop ─────────────────────────────────────────────────
    try:
        while True:
            drained = 0
            while drained < n_cams * 2:
                try:
                    cam_id, tile = frame_queue.get_nowait()
                    display.update(cam_id, tile)
                    drained += 1
                except queue.Empty:
                    break

            grid = display.render()
            cv2.imshow("Multi-Camera ReID  |  Q=quit  S=summary", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[Manager] Q pressed — shutting down …")
                break
            elif key == ord('s'):
                print_summary(workers, gallery)

    except KeyboardInterrupt:
        print("\n[Manager] Interrupted.")

    # ── Graceful shutdown ─────────────────────────────────────────────────
    for w in workers:
        w.stop()
    for w in workers:
        w.join(timeout=3.0)

    cv2.destroyAllWindows()
    print_summary(workers, gallery)
    print("[Manager] Done.\n")


if __name__ == "__main__":
    main()
