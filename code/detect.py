"""Detect fire and smoke in an image, folder, video file, or webcam.

Examples:
    python src/detect.py data/house.png
    python src/detect.py data/gen_fire.mp4 --save
    python src/detect.py 0                                    # webcam
    python src/detect.py data/ --no-show --save
    python src/detect.py data/gen_fire.mp4 --model models/best_nano_111_320.onnx

Press q to quit a video or webcam window early.
"""
import argparse
import os
import signal
import sys
from pathlib import Path

import cv2

# ultralytics otherwise retries a doomed pip install on every ONNX load (~60s a run,
# and it needs a network the Pi may not have). Must be set before ultralytics imports.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

sys.path.insert(0, str(Path(__file__).parent))
from fire_detector import Detector

ROOT = Path(__file__).parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_stop = False


def _request_stop(signum, frame):
    """Stop at the next frame so the video writer closes and the mp4 stays playable.

    Without this, Ctrl+C on a headless run kills the process mid-write and the
    file has no moov atom, i.e. the whole recording is unreadable.
    """
    global _stop
    _stop = True


def resolve_sources(source):
    """Yield (label, cv2_source, is_stream) for a webcam index, image, folder, or video."""
    if source.isdigit():
        yield f"webcam{source}", int(source), True
        return

    path = Path(source)
    if not path.exists():
        sys.exit(f"Source not found: {path}")

    if path.is_dir():
        images = sorted(f for f in path.iterdir() if f.suffix.lower() in IMG_EXT)
        if not images:
            sys.exit(f"No images found in {path}")
        for img in images:
            yield img.stem, str(img), False
    else:
        yield path.stem, str(path), path.suffix.lower() not in IMG_EXT


def run_image(detector, label, src, args, out_dir):
    frame = cv2.imread(src)
    if frame is None:
        print(f"  {label}: unreadable, skipped")
        return
    frame, detection = detector.process_frame(frame)
    print(f"  {label}: {detection or 'nothing detected'}")

    if args.save:
        dst = out_dir / f"{label}.jpg"
        cv2.imwrite(str(dst), frame)
        print(f"    saved -> {dst}")
    if args.show:
        cv2.imshow("Fire & Smoke Detection", frame)
        cv2.waitKey(0)


def run_stream(detector, label, src, args, out_dir):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    writer = None
    frames = hits = 0

    try:
        while not _stop:
            ok, frame = cap.read()
            if not ok:
                break
            frame, detection = detector.process_frame(frame)
            frames += 1
            if detection:
                hits += 1

            if args.save:
                if writer is None:
                    dst = out_dir / f"{label}.mp4"
                    h, w = frame.shape[:2]
                    # webcams often report fps as 0; 20 is a sane playback default
                    writer = cv2.VideoWriter(
                        str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps if fps > 1 else 20.0, (w, h))
                    print(f"    writing -> {dst}")
                writer.write(frame)

            if args.show:
                cv2.imshow("Fire & Smoke Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("    stopped by user")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"  {label}: {hits}/{frames} frames with fire or smoke")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="image, folder, video file, or webcam index (e.g. 0)")
    parser.add_argument("--model", default=str(ROOT / "models" / "best_nano_111.pt"),
                        help="path to a .pt or .onnx model")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="minimum confidence to draw a box (default 0.35)")
    parser.add_argument("--iou", type=float, default=0.1,
                        help="NMS IoU threshold (default 0.1)")
    parser.add_argument("--smoke-conf", type=float, default=0.5,
                        help="confidence before smoke counts as a real alert (default 0.5)")
    parser.add_argument("--save", action="store_true", help="write annotated output to outputs/")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="skip the preview window (use on a headless Pi)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    out_dir = ROOT / "outputs"
    if args.save:
        out_dir.mkdir(exist_ok=True)

    detector = Detector(args.model, iou_threshold=args.iou,
                        min_confidence=args.conf, smoke_confidence=args.smoke_conf)
    print(f"model: {args.model}  classes: {list(detector.names.values())}")

    for label, src, is_stream in resolve_sources(args.source):
        if is_stream:
            run_stream(detector, label, src, args, out_dir)
        else:
            run_image(detector, label, src, args, out_dir)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
