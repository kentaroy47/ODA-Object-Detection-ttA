"""
ODAch: An Object Detection TTA tool for Pytorch

ODAch is a test-time-augmentation (TTA) tool for 2d object detectors
with support for YOLO models from Ultralytics.

Features:
- Test Time Augmentation for object detection models
- YOLO integration (YOLOv5, YOLOv8, YOLOv9)
- Multiple augmentation strategies
- Multi-scale TTA support
- NMS and WBF integration
"""

__version__ = "0.3.0"
__author__ = "Kentaro Yoshioka"
__email__ = "meathouse47@gmail.com"

from .oda import *