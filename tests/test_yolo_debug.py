#!/usr/bin/env python3
"""
Debug script to understand why YOLO is not detecting objects
"""

import os
import sys
from PIL import Image

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_yolo_detection():
    """Debug YOLO detection with different parameters"""
    from ultralytics import YOLO
    
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    print("✓ YOLO model loaded successfully")
    
    # Load test image
    img_path = os.path.join('imgs', 'cars.jpg')
    if not os.path.exists(img_path):
        print(f"✗ Test image not found: {img_path}")
        return False
    
    print(f"Loading test image: {img_path}")
    img = Image.open(img_path)
    print(f"✓ Image loaded: {img.size}")
    
    # Test with different confidence thresholds
    thresholds = [0.1, 0.05, 0.01, 0.001]
    
    for conf in thresholds:
        print(f"\n--- Testing with confidence threshold: {conf} ---")
        
        # Run inference
        results = model(img, conf=conf, verbose=True)
        
        # Debug: Let's see what's in the results
        print(f"Results type: {type(results)}")
        print(f"Results length: {len(results)}")
        print(f"First result type: {type(results[0])}")
        print(f"First result attributes: {dir(results[0])}")
        
        # Check results - results is already a list, so we don't need [0]
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            labels = results[0].boxes.cls.cpu().numpy()
            
            print(f"✓ Detections found: {len(boxes)}")
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                print(f"  Detection {i+1}: class={int(label)}, confidence={score:.6f}, box={box}")
        else:
            print("✗ No detections found")
            print(f"Results[0].boxes: {results[0].boxes}")
    
    # Test with different image sizes
    print("\n--- Testing with different image sizes ---")
    sizes = [(640, 640), (800, 600), (1024, 768)]
    
    for size in sizes:
        print(f"\nResizing to {size}")
        resized_img = img.resize(size, Image.LANCZOS)
        
        results = model(resized_img, conf=0.01, verbose=False)
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            print(f"  Detections: {len(boxes)}")
        else:
            print("  No detections")
    
    return True
    

def test_with_different_images():
    """Test with different images in the imgs directory"""
    try:
        from ultralytics import YOLO
        
        print("\n=== Testing with different images ===")
        model = YOLO('yolov8n.pt')
        
        # List all images
        img_dir = 'imgs'
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(img_dir, img_file)
            print(f"\n--- Testing {img_file} ---")
            
            img = Image.open(img_path)
            print(f"Size: {img.size}")
            
            # Run inference with very low threshold
            results = model(img, conf=0.001, verbose=False)
            
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                labels = results.boxes.cls.cpu().numpy()
                
                print(f"Detections: {len(boxes)}")
                for i, (_, score, label) in enumerate(zip(boxes, scores, labels)):
                    print(f"  {i+1}: class={int(label)}, conf={score:.6f}")
            else:
                print("No detections")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("=== YOLO Detection Debug ===")
    
    debug_ok = debug_yolo_detection()
    images_ok = test_with_different_images()
    
    print("\n=== Summary ===")
    print(f"Debug: {'✓ PASS' if debug_ok else '✗ FAIL'}")
    print(f"Images: {'✓ PASS' if images_ok else '✗ FAIL'}") 