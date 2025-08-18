#!/usr/bin/env python3
"""
Example script demonstrating how to use the odach library with YOLOv5 and newer YOLO models
from Ultralytics.

This script shows:
1. How to wrap a YOLO model for use with odach
2. How to set up TTA (Test Time Augmentation)
3. How to run inference with TTA
"""

import sys
import os
import torch
import numpy as np

# Add the current directory to path to import odach
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from odach.oda import wrap_yolo, TTAWrapper, HorizontalFlip, VerticalFlip, Rotate90Left, Multiply


def example_basic_yolo_usage():
    """
    Basic example of using YOLO with odach
    """
    print("=== Basic YOLO Usage Example ===")
    
    # Note: In real usage, you would load an actual YOLO model like this:
    # from ultralytics import YOLO
    # yolo_model = YOLO('yolov8n.pt')  # or yolov5s.pt, etc.
    
    # For demonstration, we'll create a mock model
    class MockYOLOModel:
        def __call__(self, img, conf=0.25, iou=0.45, verbose=False):
            # Mock YOLO results
            class MockResults:
                def __init__(self):
                    self.boxes = Mock()
                    # Simulate some detections
                    self.boxes.xyxy = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]])
                    self.boxes.conf = torch.tensor([0.9, 0.8])
                    self.boxes.cls = torch.tensor([0, 1])
            
            return MockResults()
    
    # Create mock YOLO model
    yolo_model = MockYOLOModel()
    
    # Wrap the YOLO model for use with odach
    yolo_wrapper = wrap_yolo(
        model=yolo_model,
        imsize=640,
        score_threshold=0.25,
        iou_threshold=0.45
    )
    
    print(f"YOLO wrapper created with:")
    print(f"  - Image size: {yolo_wrapper.imsize}")
    print(f"  - Score threshold: {yolo_wrapper.score_threshold}")
    print(f"  - IoU threshold: {yolo_wrapper.iou_threshold}")
    
    return yolo_wrapper


def example_tta_setup(yolo_wrapper):
    """
    Example of setting up TTA with YOLO
    """
    print("\n=== TTA Setup Example ===")
    
    # Define TTA transformations
    tta_transforms = [
        HorizontalFlip(),      # Horizontal flip
        VerticalFlip(),        # Vertical flip
        Rotate90Left(),        # 90-degree left rotation
        Multiply(1.1),         # Brightness increase
        Multiply(0.9),         # Brightness decrease
    ]
    
    # Create TTA wrapper
    tta_wrapper = TTAWrapper(
        model=yolo_wrapper,
        tta=tta_transforms,
        scale=[0.8, 1.0, 1.2],  # Multiple scales
        nms="wbf",               # Use Weighted Boxes Fusion
        iou_thr=0.5,            # IoU threshold for NMS
        skip_box_thr=0.1,       # Score threshold for NMS
        weights=None             # Equal weights for all TTA variants
    )
    
    print(f"TTA wrapper created with:")
    print(f"  - Number of TTA combinations: {tta_wrapper.tta_num()}")
    print(f"  - TTA transforms: {len(tta_transforms)}")
    print(f"  - Scales: [0.8, 1.0, 1.2]")
    print(f"  - NMS method: WBF")
    
    return tta_wrapper


def example_inference_with_tta(tta_wrapper):
    """
    Example of running inference with TTA
    """
    print("\n=== Inference with TTA Example ===")
    
    # Create a mock image (in real usage, this would be your actual image)
    # Image should be in format: (channels, height, width) or (batch_size, channels, height, width)
    mock_image = torch.randn(1, 3, 640, 640)  # Single image, 3 channels, 640x640
    
    print(f"Input image shape: {mock_image.shape}")
    
    # Note: In real usage, you would run:
    # results = tta_wrapper([mock_image])  # Note: TTA wrapper expects a list of images
    
    print("Note: This is a demonstration. In real usage, you would:")
    print("1. Load actual images")
    print("2. Preprocess them (resize, normalize, etc.)")
    print("3. Run inference with TTA")
    print("4. Process the results")
    
    return tta_wrapper


def example_real_world_usage():
    """
    Example showing how this would be used in a real-world scenario
    """
    print("\n=== Real-World Usage Example ===")
    
    print("In a real-world scenario, you would use it like this:")
    print()
    print("```python")
    print("# 1. Load your YOLO model")
    print("from ultralytics import YOLO")
    print("yolo_model = YOLO('yolov8n.pt')  # or your custom trained model")
    print()
    print("# 2. Wrap it for use with odach")
    print("yolo_wrapper = wrap_yolo(")
    print("    model=yolo_model,")
    print("    imsize=640,")
    print("    score_threshold=0.25,")
    print("    iou_threshold=0.45")
    print(")")
    print()
    print("# 3. Set up TTA")
    print("tta_transforms = [HorizontalFlip(), VerticalFlip(), Rotate90Left()]")
    print("tta_wrapper = TTAWrapper(")
    print("    model=yolo_wrapper,")
    print("    tta=tta_transforms,")
    print("    scale=[0.8, 1.0, 1.2],")
    print("    nms='wbf'")
    print(")")
    print()
    print("# 4. Load and preprocess your images")
    print("from PIL import Image")
    print("import torchvision.transforms as transforms")
    print()
    print("transform = transforms.Compose([")
    print("    transforms.Resize((640, 640)),")
    print("    transforms.ToTensor(),")
    print("])")
    print()
    print("image = Image.open('your_image.jpg')")
    print("image_tensor = transform(image).unsqueeze(0)  # Add batch dimension")
    print()
    print("# 5. Run inference with TTA")
    print("results = tta_wrapper([image_tensor])")
    print()
    print("# 6. Process results")
    print("for i, result in enumerate(results):")
    print("    boxes = result['boxes']")
    print("    scores = result['scores']")
    print("    labels = result['labels']")
    print("    print(f'Image {i}: {len(boxes)} detections')")
    print("```")


def main():
    """
    Main function to run all examples
    """
    print("ODACH YOLO Integration Examples")
    print("=" * 50)
    
    try:
        # Basic YOLO usage
        yolo_wrapper = example_basic_yolo_usage()
        
        # TTA setup
        tta_wrapper = example_tta_setup(yolo_wrapper)
        
        # Inference example
        example_inference_with_tta(tta_wrapper)
        
        # Real-world usage
        example_real_world_usage()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print("\nTo run the actual tests, use:")
        print("python test_odach_yolo.py")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have all dependencies installed:")
        print("pip install torch numpy")


if __name__ == "__main__":
    main() 