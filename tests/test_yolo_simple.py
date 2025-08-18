#!/usr/bin/env python3
"""
Simple test script to verify YOLO model is working
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_yolo_basic():
    """Test basic YOLO functionality"""
    try:
        from ultralytics import YOLO
        
        print("Loading YOLO model...")
        model = YOLO('yolov8n.pt')
        print("✓ YOLO model loaded successfully")
        
        # Load test image
        img_path = os.path.join('imgs', 'cars.jpg')
        if not os.path.exists(img_path):
            print(f"✗ Test image not found: {img_path}")
            return False
        
        print(f"Loading test image: {img_path}")
        img = Image.open(img_path)
        print(f"✓ Image loaded: {img.size}")
        
        # Run inference
        print("Running YOLO inference...")
        results = model(img, conf=0.25, verbose=False)
        print("✓ Inference completed")
        
        # Check results
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy()
            
            print(f"✓ Detections found: {len(boxes)}")
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                print(f"  Detection {i+1}: class={int(label)}, confidence={score:.3f}, box={box}")
        else:
            print("✗ No detections found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_yolo_wrapper():
    """Test YOLO wrapper integration"""
    try:
        from odach.oda import wrap_yolo
        from ultralytics import YOLO
        
        print("\nTesting YOLO wrapper...")
        model = YOLO('yolov8n.pt')
        wrapper = wrap_yolo(model, imsize=640, score_threshold=0.25)
        print("✓ Wrapper created successfully")
        
        # Load and convert image
        img_path = os.path.join('imgs', 'cars.jpg')
        img = Image.open(img_path)
        img_array = np.array(img) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        
        print(f"✓ Image tensor created: {img_tensor.shape}")
        
        # Run wrapper
        print("Running wrapper inference...")
        predictions = wrapper(img_tensor)
        print("✓ Wrapper inference completed")
        
        # Check output
        print(f"Predictions: {len(predictions)} batch items")
        for i, pred in enumerate(predictions):
            print(f"  Batch {i}: {len(pred['boxes'])} detections")
            if len(pred['boxes']) > 0:
                print(f"    First detection: class={pred['labels'][0]}, confidence={pred['scores'][0]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Wrapper error: {e}")
        return False

if __name__ == "__main__":
    print("=== YOLO Basic Test ===")
    basic_ok = test_yolo_basic()
    
    print("\n=== YOLO Wrapper Test ===")
    wrapper_ok = test_yolo_wrapper()
    
    print(f"\n=== Summary ===")
    print(f"Basic YOLO: {'✓ PASS' if basic_ok else '✗ FAIL'}")
    print(f"Wrapper: {'✓ PASS' if wrapper_ok else '✗ FAIL'}")
    
    if basic_ok and wrapper_ok:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1) 