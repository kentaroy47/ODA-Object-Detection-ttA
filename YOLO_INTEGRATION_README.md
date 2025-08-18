# YOLO Integration with ODACH Library

This document explains how to use the `odach` library with YOLOv5 and newer YOLO models from [Ultralytics](https://github.com/ultralytics/ultralytics).

## Overview

The `odach` library now includes a `wrap_yolo` class that allows you to seamlessly integrate YOLO models with test-time augmentation (TTA) capabilities. This enables you to:

- Use any YOLO model (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11) with TTA
- Apply various augmentations during inference
- Use multiple scales for better detection performance
- Integrate with the existing NMS and WBF algorithms

## Features

- **Multi-version Support**: Works with YOLOv5, YOLOv8, YOLOv9, YOLOv10, and YOLOv11
- **Automatic Format Detection**: Automatically detects and handles different YOLO output formats
- **Box Normalization**: Automatically normalizes bounding box coordinates for TTA processing
- **Score Filtering**: Built-in confidence threshold filtering
- **Batch Processing**: Supports batch inference for multiple images
- **Flexible TTA**: Works with all existing odach augmentation classes

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch numpy numba ultralytics Pillow torchvision
```

## Quick Start

### 1. Basic Usage

```python
from ultralytics import YOLO
from odach.oda import wrap_yolo, TTAWrapper, HorizontalFlip, VerticalFlip

# Load your YOLO model
yolo_model = YOLO('yolov8n.pt')  # or yolov5s.pt, yolov9c.pt, etc.

# Wrap it for use with odach
yolo_wrapper = wrap_yolo(
    model=yolo_model,
    imsize=640,
    score_threshold=0.25,
    iou_threshold=0.45
)

# Set up TTA
tta_transforms = [HorizontalFlip(), VerticalFlip()]
tta_wrapper = TTAWrapper(
    model=yolo_wrapper,
    tta=tta_transforms,
    scale=[0.8, 1.0, 1.2],
    nms="wbf"
)

# Run inference with TTA
image = load_your_image()  # Your image loading function
results = tta_wrapper([image])
```

### 2. Advanced TTA Setup

```python
from odach.oda import (
    wrap_yolo, TTAWrapper, HorizontalFlip, VerticalFlip, 
    Rotate90Left, Rotate90Right, Multiply, MultiScale
)

# Create comprehensive TTA setup
tta_transforms = [
    HorizontalFlip(),      # Horizontal flip
    VerticalFlip(),        # Vertical flip
    Rotate90Left(),        # 90-degree left rotation
    Rotate90Right(),       # 90-degree right rotation
    Multiply(0.9),         # Darker image
    Multiply(1.1),         # Brighter image
]

# Multiple scales for better performance
scales = [0.75, 0.9, 1.0, 1.1, 1.25]

tta_wrapper = TTAWrapper(
    model=yolo_wrapper,
    tta=tta_transforms,
    scale=scales,
    nms="wbf",           # Use Weighted Boxes Fusion
    iou_thr=0.5,        # IoU threshold for NMS
    skip_box_thr=0.1,   # Score threshold for NMS
    weights=None         # Equal weights for all variants
)

print(f"Number of TTA combinations: {tta_wrapper.tta_num()}")
```

## API Reference

### `wrap_yolo` Class

#### Constructor

```python
wrap_yolo(
    model,              # YOLO model instance
    imsize=640,         # Input image size
    score_threshold=0.25,  # Confidence threshold
    iou_threshold=0.45     # IoU threshold for NMS
)
```

#### Parameters

- **model**: Your YOLO model instance (from `ultralytics.YOLO()`)
- **imsize**: Input image size for the model (default: 640)
- **score_threshold**: Confidence threshold for filtering detections (default: 0.25)
- **iou_threshold**: IoU threshold for NMS (default: 0.45)

#### Methods

- **`__call__(img, score_threshold=None)`**: Run inference on input images
  - **img**: Input tensor or list of tensors
  - **score_threshold**: Override default score threshold if provided
  - **Returns**: List of dictionaries with 'boxes', 'scores', and 'labels' keys

### Output Format

The wrapper converts YOLO outputs to the format expected by odach:

```python
[
    {
        'boxes': torch.Tensor,    # Normalized coordinates [0, 1]
        'scores': torch.Tensor,   # Confidence scores
        'labels': torch.Tensor     # Class labels
    },
    # ... one dict per image in batch
]
```

## Supported YOLO Versions

### YOLOv8+ (v8, v9, v10, v11)
- Output format: `results.boxes.xyxy`, `results.boxes.conf`, `results.boxes.cls`
- Automatically detected and handled

### YOLOv5
- Output format: `results.xyxy[0]` (array with [x1, y1, x2, y2, conf, cls])
- Automatically detected and handled

### Custom Models
- If your model has a different output format, the wrapper will fall back to empty results
- You can modify the wrapper to handle custom formats

## TTA Augmentation Classes

The following augmentation classes are available and work seamlessly with YOLO:

### Geometric Transformations
- **HorizontalFlip**: Horizontal image flipping
- **VerticalFlip**: Vertical image flipping
- **Rotate90Left**: 90-degree left rotation
- **Rotate90Right**: 90-degree right rotation

### Image Modifications
- **Multiply**: Brightness adjustment (0.5-1.5 range)
- **MultiScale**: Image scaling (0.5-1.5 range)

### Combined Transformations
- **MultiScaleFlip**: Scaling + horizontal flip
- **MultiScaleHFlip**: Scaling + horizontal flip (alternative)

### Composition
- **TTACompose**: Combine multiple transformations

## NMS and WBF Options

### Weighted Boxes Fusion (WBF) - Recommended
- Better performance for overlapping detections
- Handles multiple TTA variants effectively
- Default choice for most use cases

### Standard NMS
- Faster processing
- Good for real-time applications
- Less effective with multiple TTA variants

### Soft NMS
- Alternative to hard NMS
- Better for crowded scenes
- Slower than standard NMS

## Performance Considerations

### TTA Combinations
- **Monoscale**: Faster, good for most use cases
- **Multiscale**: Better accuracy, slower inference
- **Complex chains**: Exponential growth in combinations

### Memory Usage
- Each TTA variant requires additional memory
- Consider batch size and available GPU memory
- Use `torch.cuda.empty_cache()` if needed

### Inference Speed
- Base inference: ~1x speed
- With TTA: ~Nx slower (where N = number of TTA combinations)
- Use fewer TTA variants for real-time applications

## Real-World Examples

### Object Detection Competition
```python
# High-accuracy setup for competitions
tta_transforms = [
    HorizontalFlip(), VerticalFlip(),
    Rotate90Left(), Rotate90Right(),
    Multiply(0.9), Multiply(1.1)
]
scales = [0.75, 0.9, 1.0, 1.1, 1.25]

tta_wrapper = TTAWrapper(
    model=yolo_wrapper,
    tta=tta_transforms,
    scale=scales,
    nms="wbf",
    iou_thr=0.6,
    skip_box_thr=0.05
)
```

### Real-Time Application
```python
# Fast setup for real-time use
tta_transforms = [HorizontalFlip()]
scales = [1.0]  # No scaling for speed

tta_wrapper = TTAWrapper(
    model=yolo_wrapper,
    tta=tta_transforms,
    scale=scales,
    nms="nms",  # Faster NMS
    iou_thr=0.5,
    skip_box_thr=0.3
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use fewer TTA combinations
   - Use smaller image sizes

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path and imports

3. **Model Output Format Issues**
   - Verify YOLO model version
   - Check output structure with `print(results)`

4. **Performance Issues**
   - Profile with fewer TTA variants
   - Use appropriate NMS method
   - Consider model quantization

### Debug Mode

Enable verbose output to debug issues:

```python
# In YOLO inference
results = yolo_model(img, verbose=True)

# In TTA wrapper
tta_wrapper = TTAWrapper(..., verbose=True)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all YOLO tests
python test_odach_yolo.py

# Run specific test classes
python -m unittest test_odach_yolo.TestWrapYOLO -v
python -m unittest test_odach_yolo.TestYOLOTTAIntegration -v
```

## Examples

See `example_yolo_usage.py` for complete working examples.

## Contributing

To add support for new YOLO versions or custom output formats:

1. Modify the `wrap_yolo.__call__` method in `odach/oda.py`
2. Add detection logic for the new format
3. Update tests in `test_odach_yolo.py`
4. Submit a pull request

## License

This integration follows the same license as the odach library (MIT).

## References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ODACH Library](https://github.com/kentaroy47/ODA-Object-Detection-ttA)
- [Test Time Augmentation](https://arxiv.org/abs/1904.01710)
- [Weighted Boxes Fusion](https://arxiv.org/abs/1910.13302) 