"""
Pro dog detector — accuracy-first object detection (pro_detector.py).

A clean, self-contained detection pipeline tuned to catch EVERY dog in a
frame, including small / distant ones that a single low-resolution pass
misses. It is the accuracy-first counterpart to the real-time apps: use it
for analysis, validation and generating high-quality annotated footage.

Three techniques stack to maximise recall (how many real dogs are found):

  1. High-resolution inference (imgsz up to 1280) — more pixels on faraway
     dogs so the model can actually see them.
  2. Test-Time Augmentation (augment=True) — the model runs the frame at
     several scales/flips and fuses the results (higher recall).
  3. Tiled inference (SAHI-style) — the frame is sliced into overlapping
     tiles, each detected at full resolution, then merged. This is the
     standard trick for very small objects (aerial / CCTV-distance dogs).

All detections from the full frame + every tile are combined and de-duplicated
with a single class-wise Non-Max-Suppression pass, so nothing is double-counted.

FP16 half-precision runs automatically on GPU (≈2x faster, negligible accuracy
loss). yolo26x is the default for best accuracy; any YOLO26 / YOLO11 weight works.

Usage (library):
    from pro_detector import ProDogDetector
    det = ProDogDetector(model="yolo26x.pt", imgsz=960, conf=0.25, tile=True)
    result = det.detect(frame_bgr)          # {"dogs": [...], "persons": [...]}

Usage (CLI):
    # best accuracy on a video (slow, thorough):
    python pro_detector.py --source clip.mp4 --model yolo26x.pt --imgsz 1280 \
        --conf 0.20 --augment --tile --output out.mp4

    # single image:
    python pro_detector.py --source street.jpg --imgsz 960 --tile --output out.jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# COCO class ids used by stock weights
COCO_PERSON = 0
COCO_DOG = 16


# ── geometry helpers ──────────────────────────────────────────────────

def _iou(a, b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _nms(dets, iou_thresh: float):
    """Greedy Non-Max-Suppression over detections of ONE class.

    ``dets`` is a list of dicts with keys x1,y1,x2,y2,conf. Returns the kept
    subset, highest-confidence first, dropping boxes that overlap a kept box
    by more than ``iou_thresh``.
    """
    order = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep = []
    for d in order:
        box = (d["x1"], d["y1"], d["x2"], d["y2"])
        if all(_iou(box, (k["x1"], k["y1"], k["x2"], k["y2"])) < iou_thresh
               for k in keep):
            keep.append(d)
    return keep


# ── the detector ──────────────────────────────────────────────────────

class ProDogDetector:
    """Accuracy-first YOLO detector: high-res + TTA + optional tiling."""

    def __init__(self, model: str = "yolo26x.pt", conf: float = 0.25,
                 iou: float = 0.5, imgsz: int = 960, device=None, half=None,
                 augment: bool = True, tile: bool = False,
                 tile_size: int = 640, tile_overlap: float = 0.2,
                 max_det: int = 300, classes=(COCO_PERSON, COCO_DOG)):
        self.model = YOLO(model)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.augment = bool(augment)
        self.tile = bool(tile)
        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)
        self.max_det = int(max_det)
        self.classes = list(classes)

        # Resolve device once; use FP16 on GPU only (unsupported/slower on CPU).
        if device is None:
            try:
                import torch
                device = 0 if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self.half = (device != "cpu") if half is None else bool(half)

    # -- one inference call, boxes offset back to full-frame coords --------
    def _infer(self, image, ox: int = 0, oy: int = 0):
        results = self.model.predict(
            source=image, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            half=self.half, device=self.device, augment=self.augment,
            classes=self.classes, max_det=self.max_det, verbose=False,
        )
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                out.append({
                    "x1": int(x1 + ox), "y1": int(y1 + oy),
                    "x2": int(x2 + ox), "y2": int(y2 + oy),
                    "conf": float(b.conf[0]), "cls": int(b.cls[0]),
                })
        return out

    # -- slice the frame into overlapping tiles and detect in each --------
    def _tiled(self, frame):
        h, w = frame.shape[:2]
        step = max(1, int(self.tile_size * (1.0 - self.tile_overlap)))
        dets = []
        ys = list(range(0, max(1, h - self.tile_size + 1), step)) or [0]
        xs = list(range(0, max(1, w - self.tile_size + 1), step)) or [0]
        if ys[-1] != h - self.tile_size and h > self.tile_size:
            ys.append(h - self.tile_size)
        if xs[-1] != w - self.tile_size and w > self.tile_size:
            xs.append(w - self.tile_size)
        for oy in ys:
            for ox in xs:
                tile = frame[oy:oy + self.tile_size, ox:ox + self.tile_size]
                if tile.size == 0:
                    continue
                dets += self._infer(tile, ox, oy)
        return dets

    # -- public API -------------------------------------------------------
    def detect(self, frame):
        """Return {"dogs": [...], "persons": [...]} for one BGR frame.

        Each detection is {x1, y1, x2, y2, conf}. Full-frame and (optionally)
        tiled detections are merged, then de-duplicated per class via NMS.
        """
        dets = self._infer(frame)
        if self.tile:
            dets += self._tiled(frame)

        dogs = _nms([d for d in dets if d["cls"] == COCO_DOG], self.iou)
        persons = _nms([d for d in dets if d["cls"] == COCO_PERSON], self.iou)
        return {"dogs": dogs, "persons": persons}

    # -- drawing ----------------------------------------------------------
    def draw(self, frame, result):
        out = frame.copy()
        for p in result["persons"]:
            self._box(out, p, (255, 180, 60), f"person {p['conf']:.2f}")
        for d in result["dogs"]:
            self._box(out, d, (0, 220, 90), f"dog {d['conf']:.2f}")
        cv2.putText(out, f"dogs: {len(result['dogs'])}  persons: {len(result['persons'])}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return out

    @staticmethod
    def _box(img, d, color, label):
        cv2.rectangle(img, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (d["x1"], d["y1"] - th - 6),
                      (d["x1"] + tw + 4, d["y1"]), color, -1)
        cv2.putText(img, label, (d["x1"] + 2, d["y1"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)


# ── CLI ───────────────────────────────────────────────────────────────

def _run_image(det, src, out_path):
    frame = cv2.imread(src)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {src}")
    t0 = time.time()
    result = det.detect(frame)
    print(f"Detected {len(result['dogs'])} dog(s), {len(result['persons'])} "
          f"person(s) in {time.time() - t0:.2f}s")
    annotated = det.draw(frame, result)
    out_path = out_path or "pro_detected.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Saved: {out_path}")


def _run_video(det, src, out_path):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 < fps <= 120.0):
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_path:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
    n, peak = 0, 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = det.detect(frame)
        peak = max(peak, len(result["dogs"]))
        annotated = det.draw(frame, result)
        if writer:
            writer.write(annotated)
        n += 1
        if n % 20 == 0:
            print(f"  frame {n}: {len(result['dogs'])} dog(s)  "
                  f"[{n / (time.time() - t0):.1f} FPS]")
    cap.release()
    if writer:
        writer.release()
    print(f"Done — {n} frames, peak {peak} dogs in a frame, "
          f"avg {n / (time.time() - t0):.1f} FPS")
    if out_path:
        print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Accuracy-first dog detector")
    p.add_argument("--source", required=True, help="image or video path")
    p.add_argument("--model", default="yolo26x.pt", help="YOLO26/YOLO11 weights")
    p.add_argument("--imgsz", type=int, default=960,
                   help="inference size — bigger sees smaller dogs (640/960/1280)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--augment", action="store_true",
                   help="test-time augmentation (higher recall, slower)")
    p.add_argument("--tile", action="store_true",
                   help="tiled inference for very small / distant dogs")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--tile-overlap", type=float, default=0.2)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--output", default=None, help="annotated output path")
    args = p.parse_args()

    det = ProDogDetector(
        model=args.model, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        augment=args.augment, tile=args.tile, tile_size=args.tile_size,
        tile_overlap=args.tile_overlap, max_det=args.max_det)
    print(f"Model {args.model} | imgsz {args.imgsz} | conf {args.conf} | "
          f"augment {args.augment} | tile {args.tile} | device {det.device} "
          f"| fp16 {det.half}")

    src = args.source
    if str(src).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        _run_video(det, src, args.output)
    else:
        _run_image(det, src, args.output)


if __name__ == "__main__":
    main()
