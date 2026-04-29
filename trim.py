"""
create_cam2.py
==============
Simulates a cross-camera setup from a single video source.

  CAM1.mp4  →  untouched  (full video, e.g. 0:00 – 10:00)
  CAM2.mp4  →  copy of CAM1 starting from the handoff time
                (e.g. 3:00 – 10:00)

Reasoning:
  When you run camera_manager.py with both files, CAM1 plays from the start.
  CAM2 starts from the moment the person exits CAM1 and "walks into" the
  second camera — simulating a real cross-camera handoff for ReID testing.

Usage:
    python create_cam2.py                          # uses defaults below
    python create_cam2.py --input Test_Video/CAM1.mp4 --handoff 2:10
    python create_cam2.py --input Test_Video/CAM1.mp4 --handoff 3:00
"""

import cv2
import os
import sys
import argparse
import time

def parse_time(t: str) -> float:
    """Convert MM:SS or SS to seconds."""
    parts = t.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])

def fmt(sec: float) -> str:
    return f"{int(sec//60)}:{int(sec%60):02d}"

def create_cam2(input_path: str, handoff_sec: float):
    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_sec    = total_frames / fps
    handoff_frame = int(handoff_sec * fps)

    dir_name  = os.path.dirname(os.path.abspath(input_path))
    cam2_path = os.path.join(dir_name, "CAM2.mp4")

    print(f"\n{'='*56}")
    print(f"  Source     : {input_path}")
    print(f"  Resolution : {width} x {height}   FPS: {fps:.2f}")
    print(f"  Duration   : {fmt(total_sec)}  ({total_frames} frames)")
    print(f"  Handoff at : {fmt(handoff_sec)}  (frame {handoff_frame})")
    print(f"{'='*56}")
    print(f"  CAM1.mp4   : {fmt(0)} → {fmt(total_sec)}  [UNTOUCHED]")
    print(f"  CAM2.mp4   : {fmt(handoff_sec)} → {fmt(total_sec)}  [EXTRACTED]")
    print(f"{'='*56}")
    print(f"\n  Simulation: person exits CAM1 at {fmt(handoff_sec)},")
    print(f"              appears in CAM2 from frame 0 (which maps to {fmt(handoff_sec)} real time)\n")

    if handoff_frame >= total_frames:
        print("[ERROR] Handoff time is beyond the end of the video.")
        sys.exit(1)

    # Seek to handoff point and write remaining frames as CAM2
    cap.set(cv2.CAP_PROP_POS_FRAMES, handoff_frame)

    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    out      = cv2.VideoWriter(cam2_path, fourcc, fps, (width, height))
    remaining = total_frames - handoff_frame
    written   = 0
    t0        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written += 1
        if written % 300 == 0:
            pct = written / remaining * 100
            eta = ((time.time() - t0) / written) * (remaining - written)
            print(f"  Writing CAM2: {pct:5.1f}%  {written}/{remaining} frames  ETA {eta:.0f}s   ",
                  end="\r")

    cap.release()
    out.release()

    elapsed  = time.time() - t0
    cam2_sec = written / fps

    print(f"\n\n{'='*56}")
    print(f"  [DONE] CAM1.mp4  — {fmt(total_sec)} — original, not modified")
    print(f"  [DONE] CAM2.mp4  — {fmt(cam2_sec)} — saved to {cam2_path}")
    print(f"  Wrote {written} frames in {elapsed:.1f}s")
    print(f"{'='*56}")
    print(f"""
  Next steps:
  -----------
  Edit CAMERA_SOURCES in camera_manager.py:

      CAMERA_SOURCES = [
          ("Zone-1",  "Test_Video/CAM1.mp4"),
          ("Zone-2",  "Test_Video/CAM2.mp4"),
      ]

  Then run:
      python camera_manager.py

  Expected behaviour:
    • Person appears in CAM1 from 0:00
    • At {fmt(handoff_sec)} CAM1 track retires → features stored in GlobalReIDGallery
    • CAM2 starts — same person appears at frame 0
    • GlobalReIDGallery matches → same Global ID preserved  (XCAM tag)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create CAM2 as a trimmed copy of CAM1 for cross-camera ReID testing."
    )
    parser.add_argument("--input",   default="Test_Video/CAM1.mp4",
                        help="Source video (default: Test_Video/CAM1.mp4)")
    parser.add_argument("--handoff", default="2:10",
                        help="Handoff time MM:SS — when person exits CAM1 (default: 2:10)")
    args = parser.parse_args()

    create_cam2(args.input, parse_time(args.handoff))