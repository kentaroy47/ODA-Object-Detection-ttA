# ODAch, An Object Detection TTA tool for Pytorch
ODA is a test-time-augmentation (TTA) tool for 2d object detectors.

For use in Kaggle object detection competitions.

:star: if it helps you! ;)

![](imgs/res.png)

# YOLO Integration

ODAch supports YOLOv5, YOLOv8, YOLOv9, YOLOv10, and YOLOv11 models from Ultralytics!

## Quick Start with YOLO

```python
import odach as oda
from ultralytics import YOLO

# Load your YOLO model
model = YOLO('yolov8n.pt')  # or yolov5, yolov9, yolov10, yolov11

# Wrap the YOLO model for ODAch
yolo_wrapper = oda.wrap_yolo(model, imsize=640, score_threshold=0.25)

# Define TTA transformations
tta = [oda.HorizontalFlip(), oda.VerticalFlip(), oda.Rotate90Left(), oda.Rotate90Right()]

# Create TTA wrapper
tta_model = oda.TTAWrapper(yolo_wrapper, tta)

# Run inference with TTA
results = tta_model(images)
```

## YOLO Features

- **Multi-version support**: YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11
- **Automatic format conversion**: Handles YOLO output format automatically
- **Batch processing**: Process multiple images efficiently
- **Configurable thresholds**: Adjust confidence and IoU thresholds
- **Seamless integration**: Works with existing ODAch TTA pipeline

## YOLO TTA Example

```python
# Advanced YOLO TTA with multiple scales
tta = [
    oda.HorizontalFlip(),
    oda.VerticalFlip(),
    oda.Rotate90Left(),
    oda.Rotate90Right(),
    oda.Multiply(0.9),
    oda.Multiply(1.1)
]

# Multi-scale TTA
scale = [0.8, 0.9, 1.0, 1.1, 1.2]

# Create TTA wrapper with scales
tta_model = oda.TTAWrapper(yolo_wrapper, tta, scale)

# Run inference
results = tta_model(images)
```

See `example_yolo_usage.py` and `YOLO_INTEGRATION_README.md` for detailed examples.

---

# Box Format Support

ODAch supports both VOC and COCO box formats:

| Format | Description | Coordinates |
|--------|-------------|-------------|
| VOC (Pascal) | Default format | `[x1, y1, x2, y2]` |
| COCO | Alternative format | `[x, y, width, height]` |

## Using Different Box Formats

```python
import odach as oda

# VOC format (default): [x1, y1, x2, y2]
tta_model = oda.TTAWrapper(model, tta, box_format="voc")

# COCO format: [x, y, width, height]
tta_model = oda.TTAWrapper(model, tta, box_format="coco")

# Also available for wrap_yolo
yolo_wrapper = oda.wrap_yolo(model, box_format="coco")
```

## Format Conversion Utilities

```python
import odach as oda

# Convert VOC to COCO
coco_boxes = oda.voc_to_coco(voc_boxes)

# Convert COCO to VOC
voc_boxes = oda.coco_to_voc(coco_boxes)
```

---

# Install

```bash
pip install odach
```

## Development Installation

```bash
# Using uv (recommended)
uv sync --extra dev

# Run tests
uv run pytest tests/ -v
```

# Usage
See `Example.ipynb`.

The setup is very simple, similar to [ttach](https://github.com/qubvel/ttach).

## Singlescale TTA
```python
import odach as oda
# Declare TTA variations
tta = [oda.HorizontalFlip(), oda.VerticalFlip(), oda.Rotate90Left(), oda.Multiply(0.9), oda.Multiply(1.1)]

# load image
img = loadimg(impath)
# wrap model and tta
tta_model = oda.TTAWrapper(model, tta)
# Execute TTA!
boxes, scores, labels = tta_model(img)
```

## Multiscale TTA
```python
import odach as oda
# Declare TTA variations
tta = [oda.HorizontalFlip(), oda.VerticalFlip(), oda.Rotate90Left(), oda.Multiply(0.9), oda.Multiply(1.1)]
# Declare scales to tta
scale = [0.8, 0.9, 1, 1.1, 1.2]

# load image
img = loadimg(impath)
# wrap model and tta
tta_model = oda.TTAWrapper(model, tta, scale)
# Execute TTA!
boxes, scores, labels = tta_model(img)
```

* The boxes are also filtered by nms(wbf default).

* The image size should be square.

## Model Output Wrapping
* Wrap your detection model so that the output is similar to torchvision frcnn format:
`[{"boxes": [[x,y,x2,y2], ...], "labels": [0,1,..], "scores": [1.0, 0.8, ..]}]`

* Example for EfficientDets
https://www.kaggle.com/kyoshioka47/example-of-2d-single-scale-tta-with-odach/

```python
# wrap effdet
oda_effdet = oda.wrap_effdet(effdet)
# Declare TTA variations
tta = [oda.HorizontalFlip(), oda.VerticalFlip(), oda.Rotate90Left()]
# Declare scales to tta
scale = [1]
# wrap model and tta
tta_model = oda.TTAWrapper(oda_effdet, tta, scale)
```

# Examples
## YOLO TTA Examples
- `example_yolo_usage.py` - Basic YOLO integration
- `YOLO_INTEGRATION_README.md` - Detailed YOLO usage guide

## Global Wheat Detection
[Example notebook](https://www.kaggle.com/kyoshioka47/example-of-odach)

# Thanks
nms, wbf are from https://kaggle.com/zfturbo

tta is based on https://github.com/qubvel/ttach, https://github.com/andrewekhalel/edafa/tree/master/edafa and https://www.kaggle.com/shonenkov/wbf-over-tta-single-model-efficientdet
