"""
City-scale headless worker (city_worker.py) — Phase-1 scaling starter.

This is the bridge from the single-camera desktop apps (app.py … app5.py) to a
city-wide deployment. It has NO GUI. It:

  1. reads a list of cameras (name + RTSP/HTTP URL + optional lat/lon) from a
     JSON config,
  2. runs the EXISTING DogAggressionPipeline (src/inference/predict.py) on every
     camera at a sampled frame rate (dog detection needs ~3–5 FPS, not 30),
  3. on each fired alert, routes it to the NEAREST HOSPITAL (haversine) from the
     shared hospital registry (data/hospitals.json), and
  4. writes cameras, alerts and per-camera heartbeats to a SQLite database
     (data/city.db) instead of per-session JSON files — so a whole city's data
     is queryable from one place (dashboards, control room, ABC planning).

It reconnects dropped streams automatically (24/7 operation) and prints a live
one-line status per camera.

This file adds nothing to and changes nothing in the existing modules: it only
*imports* DogAggressionPipeline and reuses the same geometry/behaviour logic.

Run:
    python city_worker.py --cameras config/city_cameras.json --model yolo26n.pt --fps 4

Query the results afterwards, e.g.:
    sqlite3 data/city.db "SELECT ts,camera,behavior,risk,hospital,hospital_km FROM alerts ORDER BY ts DESC LIMIT 20;"

NOTE ON SCALE: this starter loads one pipeline per camera thread, which is fine
for a Phase-1 pilot (≈5–10 cameras on one GPU). The production scaling step is
to share one model and GPU-batch frames from many cameras per inference call —
see the city-deployment notes. Use --max-cameras to stay within VRAM.
"""

import os
import sys
import json
import math
import time
import queue
import signal
import sqlite3
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
HOSPITALS_FILE = PROJECT_ROOT / "data" / "hospitals.json"


# ── geo helpers (same maths as app5's nearest-hospital alert) ─────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def load_hospitals():
    try:
        with open(HOSPITALS_FILE, "r", encoding="utf-8") as fh:
            hs = json.load(fh)
        return hs if isinstance(hs, list) else []
    except Exception:
        return []


def nearest_hospital(lat, lon, hospitals):
    best, best_d = None, None
    for h in hospitals:
        try:
            d = haversine_km(lat, lon, float(h["lat"]), float(h["lon"]))
        except Exception:
            continue
        if best_d is None or d < best_d:
            best, best_d = h, d
    return best, best_d


# ── behaviour label (same rules as app4/app5, ported here headless) ──

def _speed_from_hist(hist, span=5):
    if len(hist) < 2:
        return 0.0
    span = min(span, len(hist))
    x0, y0, _ = hist[-span]
    x1, y1, w = hist[-1]
    w = max(1.0, w)
    denom = w * (span - 1) if span > 1 else w
    return math.hypot(x1 - x0, y1 - y0) / denom


def classify_behavior(features, risk, risk_threshold, num_dogs, sustained, spd):
    d = features.get("distance", 0.0)
    v = features.get("velocity", 0.0)
    p = features.get("posture", 0.0)
    hp = features.get("human_pose", 0.0)
    if sustained >= 1 and risk >= risk_threshold and v >= 0.4:
        label, sev = "ATTACK RISK", 3
    elif v >= 0.5 and d >= 0.4:
        label, sev = "charging at person", 3
    elif p >= 0.5 and d >= 0.35:
        label, sev = "lunging / agitated", 2
    elif v >= 0.25 and d >= 0.2:
        label, sev = "approaching person", 2
    elif d >= 0.7:
        label, sev = ("close to person (defensive human)", 2) if hp >= 0.3 \
            else ("close to person", 1)
    elif spd >= 0.09:
        label, sev = "running", 1
    elif spd >= 0.015:
        label, sev = "roaming", 0
    else:
        label, sev = "idle / resting", 0
    if num_dogs >= 3 and sev >= 1:
        label = f"pack ({num_dogs}) — {label}"
        sev = min(3, sev + 1)
    return label, sev


# ── threaded latest-frame reader (a slow pipeline never lags a feed) ──

class LatestFrameCapture:
    def __init__(self, url):
        self.url = url
        self._cap = cv2.VideoCapture(url)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._lock = threading.Lock()
        self._frame = None
        self._stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stopped:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def isOpened(self):
        return self._cap.isOpened()

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._stopped = True
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass


# ── SQLite store (replaces per-session JSON with one city-wide DB) ────

DDL = """
CREATE TABLE IF NOT EXISTS cameras (
    name TEXT PRIMARY KEY, url TEXT, lat REAL, lon REAL, added_at TEXT);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, camera TEXT, track_id INTEGER, risk REAL, behavior TEXT,
    distance REAL, velocity REAL, posture REAL, human_pose REAL,
    hospital TEXT, hospital_km REAL, hospital_phone TEXT);
CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, camera TEXT, dogs INTEGER, persons INTEGER, fps REAL, status TEXT);
CREATE INDEX IF NOT EXISTS idx_alerts_cam_ts ON alerts(camera, ts);
"""


class Store:
    """Thread-safe SQLite writer (one shared connection guarded by a lock)."""

    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(DDL)
        self._conn.commit()
        self._lock = threading.Lock()

    def upsert_camera(self, name, url, lat, lon):
        with self._lock:
            self._conn.execute(
                "INSERT INTO cameras(name,url,lat,lon,added_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET url=excluded.url, "
                "lat=excluded.lat, lon=excluded.lon",
                (name, url, lat, lon, datetime.now(timezone.utc).isoformat()))
            self._conn.commit()

    def add_alert(self, camera, a, hosp, dist):
        f = a.get("features", {})
        with self._lock:
            self._conn.execute(
                "INSERT INTO alerts(ts,camera,track_id,risk,behavior,distance,"
                "velocity,posture,human_pose,hospital,hospital_km,hospital_phone) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), camera, a["track_id"],
                 round(a["risk"], 3), a.get("behavior_label", ""),
                 f.get("distance", 0), f.get("velocity", 0), f.get("posture", 0),
                 f.get("human_pose", 0),
                 hosp.get("name", "") if hosp else "",
                 round(dist, 2) if dist is not None else None,
                 hosp.get("phone", "") if hosp else ""))
            self._conn.commit()

    def heartbeat(self, camera, dogs, persons, fps, status):
        with self._lock:
            self._conn.execute(
                "INSERT INTO heartbeats(ts,camera,dogs,persons,fps,status) "
                "VALUES(?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), camera, dogs, persons,
                 round(fps, 1), status))
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()


# ── per-camera worker ─────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    def __init__(self, cam, args, store, hospitals, stop_event):
        super().__init__(daemon=True)
        self.cam = cam
        self.args = args
        self.store = store
        self.hospitals = hospitals
        self.stop_event = stop_event
        self.name_ = cam["name"]
        self.lat = cam.get("lat")
        self.lon = cam.get("lon")
        self._hist = {}         # track_id -> deque[(cx,cy,w)] for behaviour speed
        self.status = "starting"
        self.last_dogs = self.last_persons = 0
        self.fps = 0.0

    def _make_pipeline(self):
        from src.inference.predict import DogAggressionPipeline
        pose = None if self.args.no_pose else "yolo11n-pose.pt"
        return DogAggressionPipeline(
            detector_path=self.args.model, pose_model=pose,
            det_conf=self.args.conf, risk_threshold=self.args.risk,
            sustain_frames=self.args.sustain)

    def run(self):
        try:
            pipeline = self._make_pipeline()
        except Exception as e:
            self.status = f"model-error: {e}"
            print(f"[{self.name_}] model load failed: {e}")
            return

        interval = 1.0 / max(0.5, self.args.fps)
        while not self.stop_event.is_set():
            cap = LatestFrameCapture(self.cam["url"])
            t0 = time.time()
            while not cap.isOpened() and time.time() - t0 < 8:
                time.sleep(0.2)
            if not cap.isOpened():
                self.status = "unreachable"
                self.store.heartbeat(self.name_, 0, 0, 0, "unreachable")
                print(f"[{self.name_}] unreachable — retrying in 5 s")
                cap.release()
                if self.stop_event.wait(5):
                    break
                continue

            self.status = "online"
            recent = deque()
            last_beat = 0.0
            got_frame_at = time.time()
            while not self.stop_event.is_set():
                loop_start = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    # stream stalled — reconnect if it stays empty
                    if time.time() - got_frame_at > 8:
                        self.status = "stalled"
                        print(f"[{self.name_}] stream stalled — reconnecting")
                        break
                    time.sleep(0.05)
                    continue
                got_frame_at = time.time()

                results = pipeline.process_frame(frame)
                dogs_now = len(results)
                self._label(results, dogs_now)
                persons_now = len(pipeline._last_persons)
                self.last_dogs, self.last_persons = dogs_now, persons_now

                now = time.time()
                recent.append(now)
                while recent and now - recent[0] > 2.0:
                    recent.popleft()
                self.fps = ((len(recent) - 1) / (now - recent[0])
                            if len(recent) > 1 else 0.0)

                for a in results:
                    if a.get("new_alert"):
                        self._dispatch(a)

                if now - last_beat >= self.args.heartbeat:
                    last_beat = now
                    self.store.heartbeat(self.name_, dogs_now, persons_now,
                                         self.fps, self.status)

                # pace to the target sampling FPS (don't burn the GPU flat-out)
                sleep = interval - (time.time() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)

            cap.release()
        self.status = "stopped"

    def _label(self, results, num_dogs):
        for r in results:
            tid = r["track_id"]
            cx = (r["x1"] + r["x2"]) / 2.0
            cy = (r["y1"] + r["y2"]) / 2.0
            w = max(1.0, r["x2"] - r["x1"])
            h = self._hist.setdefault(tid, deque(maxlen=6))
            h.append((cx, cy, w))
            label, _ = classify_behavior(
                r.get("features", {}), r.get("risk", 0.0), self.args.risk,
                num_dogs, r.get("sustained", 0), _speed_from_hist(h))
            r["behavior_label"] = label

    def _dispatch(self, a):
        hosp = dist = None
        if self.lat is not None and self.lon is not None and self.hospitals:
            hosp, dist = nearest_hospital(self.lat, self.lon, self.hospitals)
        self.store.add_alert(self.name_, a, hosp, dist)
        line = (f"[{self.name_}] ALERT dog#{a['track_id']} "
                f"{a.get('behavior_label','')} risk={a['risk']:.2f}")
        if hosp:
            line += (f"  → nearest hospital: {hosp['name']} ({dist:.2f} km"
                     + (f", ☎ {hosp.get('phone','')}" if hosp.get("phone") else "") + ")")
        print(line, flush=True)


# ── config + main loop ────────────────────────────────────────────────

def load_cameras(path):
    with open(path, "r", encoding="utf-8") as fh:
        cams = json.load(fh)
    if not isinstance(cams, list):
        raise ValueError("cameras file must be a JSON list of {name,url,lat,lon}")
    out = []
    for c in cams:
        if not c.get("name") or not c.get("url"):
            print(f"skip camera without name/url: {c}")
            continue
        out.append(c)
    return out


def main():
    p = argparse.ArgumentParser(description="City-scale headless stray-dog worker")
    p.add_argument("--cameras", default="config/city_cameras.json",
                   help="JSON list of {name, url, lat, lon}")
    p.add_argument("--db", default="data/city.db", help="SQLite output database")
    p.add_argument("--model", default="yolo26n.pt", help="YOLO26/YOLO11 weights")
    p.add_argument("--fps", type=float, default=4.0,
                   help="frames per second to sample PER camera (3–5 is plenty)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--risk", type=float, default=0.35)
    p.add_argument("--sustain", type=int, default=2)
    p.add_argument("--no-pose", action="store_true", help="disable the pose model (faster)")
    p.add_argument("--heartbeat", type=float, default=10.0,
                   help="seconds between per-camera status writes")
    p.add_argument("--max-cameras", type=int, default=12,
                   help="safety cap (VRAM) — one pipeline is loaded per camera")
    args = p.parse_args()

    cams = load_cameras(args.cameras)
    if len(cams) > args.max_cameras:
        print(f"{len(cams)} cameras > --max-cameras {args.max_cameras}; "
              f"processing the first {args.max_cameras}. "
              f"Raise the cap only if the GPU has the VRAM.")
        cams = cams[:args.max_cameras]

    hospitals = load_hospitals()
    store = Store(args.db)
    for c in cams:
        store.upsert_camera(c["name"], c["url"], c.get("lat"), c.get("lon"))

    print(f"City worker starting — {len(cams)} camera(s), model={args.model}, "
          f"{args.fps} FPS/cam, {len(hospitals)} hospital(s) → {args.db}")
    if not hospitals:
        print("  (no hospitals in data/hospitals.json — alerts will have no "
              "hospital routing; add some via app5's Map tab or that file)")

    stop_event = threading.Event()

    def _sig(_signum, _frame):
        print("\nstopping — finishing in-flight frames …")
        stop_event.set()
    signal.signal(signal.SIGINT, _sig)
    try:
        signal.signal(signal.SIGTERM, _sig)
    except Exception:
        pass

    workers = [CameraWorker(c, args, store, hospitals, stop_event) for c in cams]
    for w in workers:
        w.start()
        time.sleep(1.0)   # stagger model loads to smooth VRAM allocation

    # live status board
    try:
        while not stop_event.is_set():
            if stop_event.wait(5):
                break
            parts = [f"{w.name_}:{w.status}/{w.last_dogs}d/{w.fps:.1f}fps"
                     for w in workers]
            print("  " + " | ".join(parts), flush=True)
    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=3)
        store.close()
        print("stopped.")


if __name__ == "__main__":
    main()
