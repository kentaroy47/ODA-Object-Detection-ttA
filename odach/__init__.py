"""
ODAch: An Object Detection TTA tool for Pytorch

ODAch is a test-time-augmentation (TTA) tool for 2d object detectors
with support for YOLO models from Ultralytics.

Features:
- Test Time Augmentation for object detection models
- YOLO integration (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11)
- Multiple augmentation strategies
- Multi-scale TTA support
- NMS and WBF integration
- VOC/COCO box format conversion
"""

__version__ = "0.4.0"
__author__ = "Kentaro Yoshioka"
__email__ = "meathouse47@gmail.com"

from .oda import *
from .oda import voc_to_coco, coco_to_voc