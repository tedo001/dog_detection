"""
Detector - Stock COCO-pretrained YOLO11 wrapper.

Runs two models in one pass:
    - yolo11m.pt      -> person + dog bounding boxes (COCO classes 0 and 16)
    - yolo11m-pose.pt -> human keypoints (17 COCO joints) for posture features

Pose is optional; set pose_model=None to skip and save compute.
"""

from pathlib import Path

from ultralytics import YOLO


COCO_PERSON = 0
COCO_DOG = 16

# COCO keypoint indices (17-joint standard)
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16


class DogDetector:
    """Ultralytics YOLO wrapper that returns persons (w/ optional pose) + dogs."""

    def __init__(self, model_path="yolo11m.pt", pose_model="yolo11m-pose.pt",
                 conf=0.35, iou=0.45, imgsz=640, device=None, half=None,
                 secondary_model=None):
        # Open-vocabulary YOLOE models are prompted with class NAMES rather than
        # fixed COCO ids (detected by filename). Every other model loads as a
        # normal YOLO detector and behaves exactly as before.
        self.is_yoloe = "yoloe" in str(model_path).lower()
        if self.is_yoloe:
            from ultralytics import YOLOE
            self.model = YOLOE(model_path)
            names = ["dog", "person"]
            try:
                self.model.set_classes(names, self.model.get_text_pe(names))
            except Exception as e:
                print(f"[detector] YOLOE prompt setup failed ({e})")
            self._dog_ids, self._person_ids = {0}, {1}   # order of the prompt list
        else:
            self.model = YOLO(model_path)
            self._dog_ids, self._person_ids = {COCO_DOG}, {COCO_PERSON}
        self.conf = conf
        self.iou = iou
        self.imgsz = int(imgsz) if imgsz else 640

        # Pick the compute device once (GPU if available) and use FP16 there.
        # Half precision ~doubles inference throughput on CUDA with negligible
        # accuracy loss; it is unsupported / slower on CPU, so keep it off there.
        if device is None:
            try:
                import torch
                device = 0 if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self.half = (device != "cpu") if half is None else bool(half)

        self.pose = YOLO(pose_model) if pose_model else None

        # Optional SECONDARY open-vocabulary detector (YOLOE). It runs a second
        # pass prompted with "dog"/"person"; its boxes are merged with the
        # primary detector's, so dogs/humans the primary misses still get
        # caught (higher recall). Set secondary_model=None to disable.
        self.secondary = None
        if secondary_model:
            try:
                from ultralytics import YOLOE
                self.secondary = YOLOE(secondary_model)
                names = ["dog", "person"]
                self.secondary.set_classes(names, self.secondary.get_text_pe(names))
                self._sec_dog_ids, self._sec_person_ids = {0}, {1}
            except Exception as e:
                print(f"[detector] secondary YOLOE disabled ({e})")
                self.secondary = None

    def detect(self, frame):
        """
        Run detection (+ pose if enabled) on a single BGR frame.

        Returns:
            dict with:
                persons: list of {x1, y1, x2, y2, confidence, keypoints?}
                dogs:    list of {x1, y1, x2, y2, confidence}
            keypoints, when present, is a list of (x, y, visibility) tuples
            indexed by COCO joint order.
        """
        # Pass 1: detection (person + dog)
        det_kwargs = dict(
            source=frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            half=self.half, device=self.device, verbose=False,
        )
        if not self.is_yoloe:                       # YOLOE already limited to the prompt
            det_kwargs["classes"] = [COCO_PERSON, COCO_DOG]
        det_results = self.model.predict(**det_kwargs)

        persons = []
        dogs = []
        for result in det_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                det = {
                    "x1": int(box.xyxy[0][0]),
                    "y1": int(box.xyxy[0][1]),
                    "x2": int(box.xyxy[0][2]),
                    "y2": int(box.xyxy[0][3]),
                    "confidence": float(box.conf[0]),
                }
                if cls_id in self._person_ids:
                    persons.append(det)
                elif cls_id in self._dog_ids:
                    dogs.append(det)

        # Secondary detector (YOLOE): union its boxes in, then de-duplicate so
        # a dog found by both models is counted once (recall up, no doubles).
        if self.secondary is not None:
            sec_results = self.secondary.predict(
                source=frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                half=self.half, device=self.device, verbose=False)
            for result in sec_results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    det = {
                        "x1": int(box.xyxy[0][0]), "y1": int(box.xyxy[0][1]),
                        "x2": int(box.xyxy[0][2]), "y2": int(box.xyxy[0][3]),
                        "confidence": float(box.conf[0]),
                    }
                    if cls_id in self._sec_person_ids:
                        persons.append(det)
                    elif cls_id in self._sec_dog_ids:
                        dogs.append(det)
            dogs = _dedupe(dogs)
            persons = _dedupe(persons)

        # Pass 2: pose (only persons). Match keypoints to existing person boxes
        # by IoU so we keep one canonical person list.
        if self.pose is not None and persons:
            pose_results = self.pose.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                half=self.half,
                device=self.device,
                verbose=False,
            )
            for result in pose_results:
                if result.boxes is None or result.keypoints is None:
                    continue
                kp_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                for i, pbox in enumerate(boxes_xyxy):
                    pose_box = {
                        "x1": int(pbox[0]), "y1": int(pbox[1]),
                        "x2": int(pbox[2]), "y2": int(pbox[3]),
                    }
                    best = max(
                        persons,
                        key=lambda p: _iou(p, pose_box),
                        default=None,
                    )
                    if best is not None and _iou(best, pose_box) > 0.3:
                        best["keypoints"] = [
                            (float(x), float(y), float(v))
                            for x, y, v in kp_data[i]
                        ]

        return {"persons": persons, "dogs": dogs}

    def get_dog_boxes(self, frame):
        return self.detect(frame)["dogs"]

    def train(self, data_yaml, epochs=50, batch_size=16, lr=0.01,
              imgsz=640, patience=10, freeze_layers=0,
              project="experiments", name="detector", **kwargs):
        """Fine-tune on a custom dataset (optional; stock weights work fine)."""
        return self.model.train(
            data=data_yaml, epochs=epochs, batch=batch_size, lr0=lr,
            imgsz=imgsz, patience=patience, freeze=freeze_layers,
            project=project, name=name, save=True, plots=True,
            verbose=True, **kwargs,
        )

    def load_best(self, run_dir):
        best_path = Path(run_dir) / "weights" / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No best.pt at {best_path}")
        self.model = YOLO(str(best_path))


def _dedupe(dets, iou_thresh=0.6):
    """Merge overlapping boxes from the primary + secondary detectors,
    keeping the highest-confidence one (greedy NMS on dicts)."""
    order = sorted(dets, key=lambda d: d.get("confidence", 0.0), reverse=True)
    keep = []
    for d in order:
        if all(_iou(d, k) < iou_thresh for k in keep):
            keep.append(d)
    return keep


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)
