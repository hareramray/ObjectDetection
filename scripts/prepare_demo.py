"""Create teaching data and cache model/image assets before presenting."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Only create the original synthetic dataset")
    parser.add_argument("--faster-rcnn", action="store_true", help="Also cache TorchVision's pretrained detector")
    args = parser.parse_args()
    from course_utils import ensure_synthetic_dataset, audit_dataset
    data = ensure_synthetic_dataset()
    print(data)
    print(audit_dataset(data))
    if not args.offline:
        from course_utils import get_sample_image, make_demo_video, load_yolo
        print("Image:", get_sample_image())
        print("Video:", make_demo_video())
        model = load_yolo()
        result = model.predict(str(get_sample_image()), imgsz=320, device="cpu", verbose=False)[0]
        print("YOLO predictions:", len(result.boxes))
    if args.faster_rcnn:
        from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_320_fpn,
                                                  FasterRCNN_MobileNet_V3_Large_320_FPN_Weights)
        fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
        print("Faster R-CNN weights cached")

if __name__ == "__main__":
    main()
