"""
server.py — MICRO-TRACK Web Server
====================================
Headless streaming server for Lightning Studio / any display-less environment.
Models load ONLY when user clicks START in the browser UI.
Streams annotated frames via WebSocket to frontend.html at port 8005.

Usage:
    python server.py
    # then open http://localhost:8005 (or Lightning Studio Ports tab)
"""

import asyncio
import base64
import json
import queue
import threading
import time
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Global state ────────────────────────────────────────────────────────────
_pipeline_running = False
_workers          = []
_gallery          = None
_frame_queue      = queue.Queue(maxsize=60)
_log_messages     = []
_log_lock         = threading.Lock()
_frame_clients    = []
_log_clients      = []
_clients_lock     = threading.Lock()

# Shared model handles (loaded once on first START)
_yolo_handle = None
_reid_handle = None


# ══════════════════════════════════════════════════════════════════════════════
# Logging bridge — isatty-safe, forwards prints to WS log panel
# ══════════════════════════════════════════════════════════════════════════════

class _LogBridge:
    def __init__(self, orig):
        self._orig = orig

    def write(self, msg):
        try:
            self._orig.write(msg)
        except Exception:
            pass
        if msg and msg.strip():
            with _log_lock:
                _log_messages.append(msg.strip())
                if len(_log_messages) > 500:
                    _log_messages.pop(0)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:
            return False

    def fileno(self):
        try:
            return self._orig.fileno()
        except Exception:
            raise AttributeError("fileno")

    def __getattr__(self, name):
        return getattr(self._orig, name)

# Patch stdout BEFORE uvicorn configures logging
_real_stdout = sys.stdout
sys.stdout   = _LogBridge(_real_stdout)


# ══════════════════════════════════════════════════════════════════════════════
# Broadcast tasks
# ══════════════════════════════════════════════════════════════════════════════

async def _broadcast_frames():
    while True:
        try:
            msg = _frame_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        dead = []
        with _clients_lock:
            clients = list(_frame_clients)
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        with _clients_lock:
            for ws in dead:
                if ws in _frame_clients:
                    _frame_clients.remove(ws)


async def _broadcast_logs():
    last_idx = 0
    while True:
        await asyncio.sleep(0.4)
        with _log_lock:
            new = _log_messages[last_idx:]
            last_idx = len(_log_messages)
        if not new:
            continue
        payload = json.dumps({"logs": new})
        dead = []
        with _clients_lock:
            clients = list(_log_clients)
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        with _clients_lock:
            for ws in dead:
                if ws in _log_clients:
                    _log_clients.remove(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_broadcast_frames())
    asyncio.create_task(_broadcast_logs())
    yield


app = FastAPI(title="MICRO-TRACK", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline — starts in a daemon thread when /api/start is called
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(config: dict):
    global _pipeline_running, _workers, _gallery, _frame_queue
    global _yolo_handle, _reid_handle

    try:
        from ultralytics import YOLO
        from reid_test import (
            ReIDExtractor, GlobalReIDGallery, PersonTracker, draw_frame,
            CONF_THRESH, IOU_THRESH, REID_THRESH, CROSS_CAM_THRESH,
            MAX_LOST_FRAMES, REENTRY_WINDOW_SECONDS,
        )

        sources     = config.get("sources", [])
        yolo_path   = config.get("yolo_model",  "yolov8m.pt")
        reid_path   = config.get("reid_model",  "onnx_model/resnet50_reid.onnx")
        retention   = float(config.get("retention_seconds", REENTRY_WINDOW_SECONDS))
        conf_thresh = float(config.get("conf_thresh", CONF_THRESH))

        # Load models once and cache them
        if _yolo_handle is None:
            print(f"[Server] Loading YOLO: {yolo_path}")
            _yolo_handle = YOLO(yolo_path)
            print("[Server] YOLO ready")

        if _reid_handle is None:
            print(f"[Server] Loading ReID: {reid_path}")
            _reid_handle = ReIDExtractor(reid_path)
            print("[Server] ReID ready")

        _gallery = GlobalReIDGallery(
            cross_cam_thresh       = CROSS_CAM_THRESH,
            reentry_window_seconds = retention,
        )
        _frame_queue = queue.Queue(maxsize=60)
        _workers     = []

        for cam_id, src in enumerate(sources):
            label = src.get("label", f"Zone-{cam_id+1}")
            url   = src.get("url",   str(cam_id))

            from reid_test import (IOU_THRESH, REID_THRESH,
                                   MAX_LOST_FRAMES, REENTRY_WINDOW_SECONDS)
            tracker = PersonTracker(
                reid                   = _reid_handle,
                global_gallery         = _gallery,
                cam_id                 = cam_id,
                iou_thresh             = IOU_THRESH,
                reid_thresh            = REID_THRESH,
                max_lost_frames        = MAX_LOST_FRAMES,
                reentry_window_seconds = retention,
            )
            w = {"cam_id": cam_id, "label": label, "url": url,
                 "fps": 0.0, "active": 0, "connected": False, "tracker": tracker}
            _workers.append(w)

            t = threading.Thread(
                target  = _camera_worker,
                args    = (cam_id, label, url, conf_thresh),
                daemon  = True,
                name    = f"Cam-{cam_id}",
            )
            w["thread"] = t
            t.start()
            print(f"[Server] Started: {label} → {url}")

        print(f"[Server] {len(sources)} camera(s) running")

    except Exception as e:
        traceback.print_exc()
        _pipeline_running = False
        print(f"[Server] Pipeline FAILED: {e}")


def _camera_worker(cam_id, label, url, conf_thresh):
    global _pipeline_running

    source    = int(url) if str(url).isdigit() else url
    RECONNECT = 3.0

    # import inside thread to avoid any main-thread issue
    from reid_test import draw_frame

    while _pipeline_running:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            _set_worker(cam_id, connected=False)
            _push_blank(cam_id, label, "NO SIGNAL")
            time.sleep(RECONNECT)
            continue

        _set_worker(cam_id, connected=True)
        print(f"[Cam {cam_id}|{label}] Connected ✓")

        tracker = next((w["tracker"] for w in _workers if w["cam_id"] == cam_id), None)
        fps_count = 0
        fps_start = time.time()
        fps_val   = 0.0

        while _pipeline_running:
            ret, frame = cap.read()
            if not ret:
                print(f"[Cam {cam_id}|{label}] Stream ended / lost")
                _set_worker(cam_id, connected=False)
                break

            # ── YOLO ──────────────────────────────────────────────────────
            yolo_out   = _yolo_handle(frame, classes=[0],
                                      conf=conf_thresh, verbose=False)[0]
            detections = []
            if yolo_out.boxes is not None:
                for box in yolo_out.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, float(box.conf[0])])

            # ── ReID tracker ───────────────────────────────────────────────
            results = tracker.update(frame, detections) if tracker else []

            # ── FPS ────────────────────────────────────────────────────────
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_val   = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()
                _set_worker(cam_id, fps=round(fps_val, 1),
                            active=sum(1 for t in (tracker.tracks.values()
                                       if tracker else []) if t.confirmed))

            # ── Annotate ───────────────────────────────────────────────────
            ann = draw_frame(frame.copy(), results,
                             tracker, fps_val, cam_label=f"[{label}]")

            # Resize to 640×360 for streaming
            h, w = ann.shape[:2]
            scale = min(640/w, 360/h)
            if scale < 1.0:
                ann = cv2.resize(ann, (int(w*scale), int(h*scale)),
                                 interpolation=cv2.INTER_AREA)

            _, jpeg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(jpeg.tobytes()).decode()

            msg = json.dumps({"cam_id": cam_id, "label": label,
                              "frame": b64, "fps": round(fps_val, 1)})
            try:
                _frame_queue.put_nowait(msg)
            except queue.Full:
                try:
                    _frame_queue.get_nowait()   # drop oldest frame
                    _frame_queue.put_nowait(msg)
                except Exception:
                    pass

        cap.release()
        if _pipeline_running:
            print(f"[Cam {cam_id}|{label}] Reconnecting in {RECONNECT}s…")
            time.sleep(RECONNECT)

    print(f"[Cam {cam_id}|{label}] Worker stopped")


def _set_worker(cam_id, **kw):
    for w in _workers:
        if w["cam_id"] == cam_id:
            w.update(kw)


def _push_blank(cam_id, label, msg):
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, label, (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 200, 80), 2)
    cv2.putText(img, msg,   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 200), 2)
    _, jpeg = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(jpeg.tobytes()).decode()
    try:
        _frame_queue.put_nowait(json.dumps(
            {"cam_id": cam_id, "label": label, "frame": b64, "fps": 0}))
    except queue.Full:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# REST endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def api_status():
    gallery_info = {}
    if _gallery:
        try:
            gallery_info = _gallery.summary()
        except Exception:
            pass
    return JSONResponse({
        "running": _pipeline_running,
        "workers": [
            {"cam_id":    w["cam_id"],
             "label":     w["label"],
             "url":       w["url"],
             "fps":       w.get("fps", 0),
             "active":    w.get("active", 0),
             "connected": w.get("connected", False)}
            for w in _workers
        ],
        "gallery": gallery_info,
    })


@app.post("/api/start")
async def api_start(config: dict):
    global _pipeline_running
    if _pipeline_running:
        return JSONResponse({"ok": False, "msg": "Already running"})
    _pipeline_running = True
    threading.Thread(target=_run_pipeline, args=(config,),
                     daemon=True, name="PipelineInit").start()
    return JSONResponse({"ok": True, "msg": "Pipeline starting…"})


@app.post("/api/stop")
async def api_stop():
    global _pipeline_running
    _pipeline_running = False
    print("[Server] Pipeline stopped by user")
    return JSONResponse({"ok": True, "msg": "Stopped"})


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    await ws.accept()
    with _clients_lock:
        _frame_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with _clients_lock:
            if ws in _frame_clients:
                _frame_clients.remove(ws)


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    with _clients_lock:
        _log_clients.append(ws)
    with _log_lock:
        if _log_messages:
            await ws.send_text(json.dumps({"logs": _log_messages[-100:]}))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with _clients_lock:
            if ws in _log_clients:
                _log_clients.remove(ws)


# ══════════════════════════════════════════════════════════════════════════════
# Frontend
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "frontend.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse(
        "<h1>frontend.html not found</h1><p>Place frontend.html next to server.py</p>",
        status_code=404)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*52)
    print("  MICRO-TRACK Web Server")
    print("  http://0.0.0.0:8005")
    print("  Open in browser → http://localhost:8005")
    print("="*52 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="warning")