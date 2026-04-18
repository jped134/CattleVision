"""CattleVision CLI entry point.

Usage examples
--------------
Identify cows in a single image (print results to stdout):
    python detect.py --image path/to/frame.jpg

Process a video and write annotated output:
    python detect.py --video herd.mp4 --output annotated.mp4

Enroll a known cow into the identity database:
    python detect.py --enroll cow_007 --image reference.jpg --db gallery.npz

Train the re-ID embedder from scratch:
    python detect.py --train --config config/default.yaml

All settings can be overridden via --config <yaml file>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CattleVision – individual cow identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",  metavar="PATH", help="Identify cows in an image file")
    mode.add_argument("--video",  metavar="PATH", help="Identify cows in a video file")
    mode.add_argument("--train",  action="store_true", help="Train the re-ID embedder")
    mode.add_argument("--enroll", metavar="COW_ID", help="Enroll a cow into the gallery")

    p.add_argument("--config",    metavar="PATH", default="config/default.yaml",
                   help="Path to YAML config (default: config/default.yaml)")
    p.add_argument("--embedder",  metavar="PATH", default=None,
                   help="Path to trained embedder checkpoint")
    p.add_argument("--db",        metavar="PATH", default=None,
                   help="Path to identity database (.npz)")
    p.add_argument("--output",    metavar="PATH", default=None,
                   help="Where to write annotated video output")
    p.add_argument("--show",      action="store_true",
                   help="Display live video in a window while processing")
    return p.parse_args(argv)


def _load_cfg(args):
    config_path = Path(args.config)
    if config_path.exists():
        from cattlevision.utils.config import load_config
        return load_config(config_path)
    # Return a minimal stub so the code below doesn't crash
    from cattlevision.utils.config import Config
    return Config({})


def cmd_identify_image(args, cfg):
    import cv2
    from cattlevision import CowIdentifier

    identifier = CowIdentifier(
        embedder_path=args.embedder,
        database_path=args.db,
        similarity_threshold=cfg.get("database", {}).get("similarity_threshold", 0.6),
        det_conf_threshold=cfg.get("detector", {}).get("conf_threshold", 0.25),
    )
    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Error: cannot read image '{args.image}'")

    results = identifier.identify_image(image)
    if not results:
        print("No cows detected.")
        return

    print(f"Detected {len(results)} cow(s):")
    for i, r in enumerate(results, 1):
        print(
            f"  [{i}] id={r.cow_id}  similarity={r.similarity:.3f}"
            f"  det_conf={r.confidence:.3f}  bbox={r.bbox.tolist()}"
        )

    if args.output:
        annotated = identifier._annotate(image, results)
        cv2.imwrite(args.output, annotated)
        print(f"Annotated image saved to {args.output}")


def cmd_identify_video(args, cfg):
    import time
    import cv2
    from cattlevision import CowIdentifier
    from cattlevision.pipeline.tracker import CowTracker
    from cattlevision.behavioral.monitor import BehavioralMonitor
    from cattlevision.utils.visualization import draw_identities

    identifier = CowIdentifier(
        embedder_path=args.embedder,
        database_path=args.db,
        similarity_threshold=cfg.get("database", {}).get("similarity_threshold", 0.6),
        det_conf_threshold=cfg.get("detector", {}).get("conf_threshold", 0.25),
    )
    t_cfg = cfg.get("tracker", {})
    tracker = CowTracker(
        max_age=t_cfg.get("max_age", 30),
        min_hits=t_cfg.get("min_hits", 3),
        iou_threshold=t_cfg.get("iou_threshold", 0.30),
        appearance_weight=t_cfg.get("appearance_weight", 0.40),
    )

    b_cfg = cfg.get("behavioral", {})
    monitor = BehavioralMonitor.from_config(b_cfg)
    behavioral_enabled = b_cfg.get("enabled", True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Error: cannot open video '{args.video}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))

    frame_count = 0
    total_alerts = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            identify_results = identifier.identify_image(frame)
            tracker_results  = tracker.update(identify_results)

            alerts = []
            if behavioral_enabled:
                alerts = monitor.process(
                    tracker_results=tracker_results,
                    tracks=tracker._tracks,
                    frame=frame,
                    frame_number=frame_count,
                    timestamp=time.time(),
                    frame_wh=(w, h),
                )

            for alert in alerts:
                total_alerts += 1
                print(alert)

            if writer or args.show:
                # Draw tracker boxes (including behavioral alert highlights)
                alerted_ids = {a.cow_id for a in alerts}
                bboxes  = [tr["bbox"] for tr in tracker_results if tr.get("confirmed")]
                cow_ids = [tr["cow_id"] for tr in tracker_results if tr.get("confirmed")]
                annotated = draw_identities(frame, bboxes, cow_ids)
                # Overlay alert markers
                for alert in alerts:
                    x1, y1, x2, y2 = alert.bbox.astype(int)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        annotated,
                        f"ALERT:{alert.anomaly_subtype}({alert.score:.2f})",
                        (x1, max(0, y1 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
                    )
                if writer:
                    writer.write(annotated)
                if args.show:
                    cv2.imshow("CattleVision – Behavioral Monitor", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames. Behavioral alerts: {total_alerts}")
    if behavioral_enabled:
        summary = monitor.profiles_summary()
        print(f"Cow profiles: {len(summary)} individual(s) tracked.")


def cmd_enroll(args, cfg):
    import cv2
    from cattlevision import CowIdentifier
    from pathlib import Path

    db_path = args.db or "gallery.npz"
    identifier = CowIdentifier(
        embedder_path=args.embedder,
        database_path=db_path if Path(db_path).exists() else None,
    )
    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Error: cannot read image '{args.image}'")

    identifier.enroll(args.enroll, image)
    identifier.save_database(db_path)
    print(f"Enrolled '{args.enroll}' → saved database to {db_path}")
    print(f"Database now contains {identifier.database.num_identities} identity/identities.")


def cmd_train(args, cfg):
    from cattlevision.training.trainer import Trainer

    t_cfg = cfg.get("training", {})
    trainer = Trainer(
        data_root   = t_cfg.get("data_root",    "data/reid"),
        output_dir  = t_cfg.get("output_dir",   "runs/train"),
        embedding_dim = t_cfg.get("embedding_dim", 256),
        epochs      = t_cfg.get("epochs",       50),
        batch_size  = t_cfg.get("batch_size",   64),
        lr          = t_cfg.get("lr",           3e-4),
        margin      = t_cfg.get("margin",       0.3),
        val_split   = t_cfg.get("val_split",    0.15),
        device      = t_cfg.get("device",       None),
    )
    trainer.run()


def main(argv=None):
    args = _parse_args(argv)
    cfg  = _load_cfg(args)

    if args.train:
        cmd_train(args, cfg)
    elif args.enroll:
        cmd_enroll(args, cfg)
    elif args.image:
        cmd_identify_image(args, cfg)
    else:
        cmd_identify_video(args, cfg)


if __name__ == "__main__":
    main()
