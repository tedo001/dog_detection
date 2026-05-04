"""
Dog Aggression Detection - Main Training Entry Point

Usage:
    # Full pipeline (prepare data → train detector → train classifier)
    python train.py

    # Individual stages
    python train.py --stage prepare
    python train.py --stage detector
    python train.py --stage classifier
    # AI Agent mode (auto-tune hyperparameters)
    python train.py --stage agent
    python train.py --stage agent --max-experiments 10

    # Inference
    python train.py --stage predict --source path/to/image.jpg

    # Evaluate
    python train.py --stage evaluate
"""

import argparse
import sys
from pathlib import Path
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_prepare(config):
    """Stage 0: Prepare data - validate, split, crop."""
    from src.data.prepare import (
        load_config as load_data_config,
        validate_dataset,
        create_splits,
        crop_detections,
        print_dataset_summary,
    )

    data_cfg = load_data_config(config["data"]["dataset_config"])
    data_root = data_cfg["path"]

    print("\n[train.py] STAGE 0: DATA PREPARATION\n")

    report = print_dataset_summary(data_root)

    if not report["valid"]:
        print("\nFATAL: Dataset validation failed. Fix issues above.")
        sys.exit(1)

    create_splits(data_root)
    crop_detections(data_root, output_dir=f"{data_root}/crops")
    print("\n[train.py] Data preparation complete!\n")


def run_detector(config):
    """Stage 1: Train dog detector."""
    from src.training.train_detector import train_detector
    return train_detector(config)


def run_classifier(config):
    """Stage 2: Train aggression classifier."""
    from src.training.train_classifier import train_classifier
    return train_classifier(config)


def run_agent(config, max_experiments=None):
    """AI Agent: Auto-tune training with experiment automation."""
    from src.agent.training_agent import TrainingAgent

    agent = TrainingAgent(config_path="config/config.yaml")
    if max_experiments:
        agent.max_experiments = max_experiments
    agent.run(stage="classifier")


def run_evaluate(config):
    """Evaluate trained models."""
    from src.evaluation.evaluate import (
        evaluate_detector,
        evaluate_classifier,
        save_evaluation_report,
    )
    from src.models.detector import DogDetector
    from src.models.classifier import AggressionClassifier
    from src.data.prepare import get_dataloaders, load_config as load_data_config

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = load_data_config(config["data"]["dataset_config"])

    print("\n[train.py] EVALUATION\n")

    all_metrics = {}

    # Evaluate detector if weights exist
    det_weights = Path("experiments") / "detector_best" / "weights" / "best.pt"
    if det_weights.exists():
        detector = DogDetector(model_path=str(det_weights))
        all_metrics["detector"] = evaluate_detector(
            detector, config["data"]["dataset_config"]
        )

    # Evaluate classifier if weights exist
    cls_weights = Path("experiments") / "classifier_best" / "checkpoints" / "best.pt"
    if cls_weights.exists():
        classifier = AggressionClassifier.load(str(cls_weights), device=str(device))
        crop_dir = Path(data_cfg["path"]) / "crops"
        _, val_loader = get_dataloaders(str(crop_dir))
        all_metrics["classifier"] = evaluate_classifier(
            classifier, val_loader, device
        )

    if all_metrics:
        save_evaluation_report(all_metrics, "experiments/evaluation_report.json")
    else:
        print("  No trained models found. Run training first.")


def run_predict(config, source, output=None):
    """Run inference on image/video using geometric risk rules."""
    from src.inference.predict import DogAggressionPipeline

    detector_path = config.get("inference", {}).get(
        "detector_path", config["detector"]["model"]
    )
    risk_threshold = config["inference"].get(
        "risk_threshold", config["inference"].get("aggression_threshold", 0.6)
    )

    inf = config["inference"]
    pipeline = DogAggressionPipeline(
        detector_path=detector_path,
        pose_model=inf.get("pose_model", "yolo11m-pose.pt"),
        det_conf=inf["detection_conf"],
        risk_threshold=risk_threshold,
        smoothing_alpha=inf.get("smoothing_alpha", 0.35),
        sustain_frames=inf.get("sustain_frames", 5),
        cooldown_frames=inf.get("cooldown_frames", 30),
    )

    if source == "webcam":
        pipeline.predict_webcam()
    elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        pipeline.predict_video(source, output_path=output)
    else:
        results, _ = pipeline.predict_image(source, save_path=output)
        for r in results:
            status = "ALERT" if r["alert"] else "safe"
            print(f"  Dog#{r['track_id']}: {status} (risk: {r['risk']:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description="Dog Aggression Detection Training Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "prepare", "detector", "classifier", "agent", "evaluate", "predict"],
        help="Which stage to run",
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--source", type=str, default=None, help="Image/video for prediction")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--max-experiments", type=int, default=None, help="Agent max experiments")
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n" + "=" * 60)
    print("  DOG AGGRESSION DETECTION SYSTEM")
    print(f"  Project: {config['project']['name']}")
    print("=" * 60)

    if args.stage == "all":
        run_prepare(config)
        run_detector(config)
        run_classifier(config)

    elif args.stage == "prepare":
        run_prepare(config)
    elif args.stage == "detector":
        run_detector(config)
    elif args.stage == "classifier":
        run_classifier(config)
    elif args.stage == "agent":
        run_agent(config, max_experiments=args.max_experiments)
    elif args.stage == "evaluate":
        run_evaluate(config)
    elif args.stage == "predict":
        if not args.source:
            print("ERROR: --source required for prediction")
            sys.exit(1)
        run_predict(config, args.source, args.output)


if __name__ == "__main__":
    main()
