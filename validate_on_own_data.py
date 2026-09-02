"""
Measure detector accuracy on the project's OWN corpus, not on FGVD.

The headline 0.892 mAP50 was measured on FGVD's validation split - 884 still
images from a published dataset. This script measures the same detector against
manually corrected annotations drawn from the 124 traffic videos the pipeline
actually runs on, which is the number that should be quoted for the deployment
domain.

Run make_validation_set.py first, correct the labels by hand, then:

    python src/validate_on_own_data.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline_core import PATHS, DETECTION  # noqa: E402

DATA = os.path.join("datasets", "own_validation", "dataset.yaml")


def main():
    if not os.path.exists(DATA):
        print(f"{DATA} not found - run make_validation_set.py first")
        return

    from ultralytics import YOLO
    model = YOLO(PATHS["yolo_weights"])

    r = model.val(data=DATA, split="val", imgsz=DETECTION["imgsz"],
                  batch=8, verbose=True, plots=True,
                  project="outputs/detector_val", name="own_corpus",
                  exist_ok=True)

    print()
    print("=" * 66)
    print("DETECTOR PERFORMANCE ON THE PROJECT'S OWN CORPUS")
    print("=" * 66)
    names = model.names
    print(f"{'class':18s} {'mAP50':>9s} {'mAP50-95':>10s} {'precision':>11s} {'recall':>8s}")
    for i, c in enumerate(r.box.ap_class_index):
        print(f"{names[int(c)]:18s} {r.box.ap50[i]:9.4f} {r.box.ap[i]:10.4f} "
              f"{r.box.p[i]:11.4f} {r.box.r[i]:8.4f}")
    print(f"{'ALL':18s} {r.box.map50:9.4f} {r.box.map:10.4f} "
          f"{r.box.mp:11.4f} {r.box.mr:8.4f}")
    print()
    print("compare with FGVD validation split: mAP50 0.8917, mAP50-95 0.7220")
    print(f"difference on own corpus         : {r.box.map50 - 0.8917:+.4f} mAP50")
    print()
    print("A drop here is expected and worth reporting - FGVD is curated still")
    print("imagery, this corpus is real video at four resolutions. The point is")
    print("to state the deployment-domain figure rather than assume it transfers.")


if __name__ == "__main__":
    main()
