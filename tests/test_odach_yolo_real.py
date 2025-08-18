#!/usr/bin/env python3
"""
Real YOLO model integration tests for the odach library
Tests the wrap_yolo class with actual YOLO models from ultralytics
"""

import unittest
import numpy as np
import torch
import sys
import os
import shutil
from PIL import Image

# Add the current directory to path to import odach
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from odach.oda import (
    wrap_yolo, TTAWrapper, HorizontalFlip, VerticalFlip
)
from odach.nms import nms
from odach.wbf import weighted_boxes_fusion


class TestRealYOLOIntegration(unittest.TestCase):
    """Test odach integration with real YOLO models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with real YOLO model"""
        try:
            from ultralytics import YOLO
            
            # Download a small YOLO model for testing
            cls.yolo_model = YOLO('yolov8n.pt')  # nano model (smallest)
            cls.yolo_available = True
            
            # Create test image
            cls.test_image = cls._create_test_image()
            cls.test_image_tensor = cls._image_to_tensor(cls.test_image)
            
        except ImportError:
            print("Warning: ultralytics not available, skipping real YOLO tests")
            cls.yolo_available = False
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            cls.yolo_available = False
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_test_image(cls):
        """Load existing car image for testing"""
        # Load existing car image from imgs directory
        img_path = os.path.join(os.path.dirname(__file__), 'imgs', 'cars.jpg')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            # Resize to standard size for testing
            try:
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
            except AttributeError:
                # Fallback for older Pillow versions
                img = img.resize((640, 480), Image.LANCZOS)
            return img
        else:
            # Fallback to created image if file doesn't exist
            img = Image.new('RGB', (640, 480), color='lightblue')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            
            # Draw a car-like shape (red rectangle with wheels)
            draw.rectangle([100, 200, 300, 280], fill='red')
            draw.ellipse([120, 270, 160, 310], fill='black')  # wheel
            draw.ellipse([240, 270, 280, 310], fill='black')  # wheel
            
            return img
    
    @classmethod
    def _image_to_tensor(cls, pil_image):
        """Convert PIL image to tensor format"""
        # Convert PIL to numpy array and normalize
        img_array = np.array(pil_image) / 255.0
        
        # Convert to tensor and add batch dimension
        # Shape: [height, width, channels] -> [channels, height, width] -> [batch, channels, height, width]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        
        return img_tensor
    
    def setUp(self):
        """Set up for each test"""
        if not self.yolo_available:
            self.skipTest("YOLO model not available")
        
        # Create YOLO wrapper
        self.yolo_wrapper = wrap_yolo(self.yolo_model, imsize=640, score_threshold=0.25)
    
    def test_real_yolo_inference(self):
        """Test real YOLO model inference through wrapper"""
        # Run inference with real YOLO model
        predictions = self.yolo_wrapper(self.test_image_tensor)
        
        # Check output format
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)  # batch size
        
        pred = predictions[0]
        self.assertIn('boxes', pred)
        self.assertIn('scores', pred)
        self.assertIn('labels', pred)
        
        # Check data types
        self.assertIsInstance(pred['boxes'], torch.Tensor)
        self.assertIsInstance(pred['scores'], torch.Tensor)
        self.assertIsInstance(pred['labels'], torch.Tensor)
        
        # Check that we got some detections (should detect the colored rectangles)
        self.assertGreater(len(pred['boxes']), 0)
        
        # Check box format (should be normalized to [0, 1])
        if len(pred['boxes']) > 0:
            boxes = pred['boxes'].numpy()
            self.assertTrue(np.all(boxes >= 0))
            self.assertTrue(np.all(boxes <= 1))
    
    def test_real_yolo_score_filtering(self):
        """Test score threshold filtering with real YOLO model"""
        # Test with high threshold (should filter out low confidence detections)
        high_threshold_wrapper = wrap_yolo(self.yolo_model, imsize=640, score_threshold=0.8)
        
        predictions_high = high_threshold_wrapper(self.test_image_tensor)
        predictions_low = self.yolo_wrapper(self.test_image_tensor)
        
        # High threshold should have fewer or equal detections
        self.assertLessEqual(len(predictions_high[0]['scores']), len(predictions_low[0]['scores']))
    
    def test_real_yolo_batch_processing(self):
        """Test batch processing with real YOLO model"""
        # Create batch tensor instead of list of tensors
        batch_tensor = torch.cat([self.test_image_tensor, self.test_image_tensor], dim=0)
        
        # Run batch inference
        predictions = self.yolo_wrapper(batch_tensor)
        
        # Check batch output
        self.assertEqual(len(predictions), 2)
        
        # Both should have similar detections
        for pred in predictions:
            self.assertIn('boxes', pred)
            self.assertIn('scores', pred)
            self.assertIn('labels', pred)
    
    def test_real_yolo_tta_integration(self):
        """Test TTA integration with real YOLO model"""
        # Create TTA transforms
        tta_transforms = [HorizontalFlip(), VerticalFlip()]
        
        # Create TTA wrapper
        tta_wrapper = TTAWrapper(
            model=self.yolo_wrapper,
            tta=tta_transforms,
            scale=[1],
            nms="wbf",
            iou_thr=0.5,
            skip_box_thr=0.1
        )
        
        # Run TTA inference
        predictions = tta_wrapper(self.test_image_tensor)
        
        # Check TTA output
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)
        
        # Check first prediction
        pred = predictions[0]
        self.assertIn('boxes', pred)
        self.assertIn('scores', pred)
        self.assertIn('labels', pred)
        
        # TTA should produce results
        if len(pred['boxes']) > 0:
            self.assertIsInstance(pred['boxes'], torch.Tensor)
            self.assertIsInstance(pred['scores'], torch.Tensor)
            self.assertIsInstance(pred['labels'], torch.Tensor)
    
    def test_real_yolo_multiscale_tta(self):
        """Test multiscale TTA with real YOLO model"""
        # Create TTA wrapper with multiple scales
        tta_transforms = [HorizontalFlip()]
        scales = [0.8, 1.0, 1.2]
        
        tta_wrapper = TTAWrapper(
            model=self.yolo_wrapper,
            tta=tta_transforms,
            scale=scales,
            nms="wbf"
        )
        
        # Check TTA combinations
        expected_tta_count = len(scales) * (2 ** len(tta_transforms))
        self.assertEqual(tta_wrapper.tta_num(), expected_tta_count)
        
        # Run multiscale TTA
        predictions = tta_wrapper(self.test_image_tensor)
        
        # Should produce results
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)
        
        pred = predictions[0]
        self.assertIn('boxes', pred)
        self.assertIn('scores', pred)
        self.assertIn('labels', pred)
    
    def test_real_yolo_nms_integration(self):
        """Test NMS integration with real YOLO outputs"""
        # Run YOLO inference multiple times to get different predictions
        predictions1 = self.yolo_wrapper(self.test_image_tensor)
        predictions2 = self.yolo_wrapper(self.test_image_tensor)
        
        # Extract boxes, scores, and labels
        boxes_list = [
            predictions1[0]['boxes'].numpy(),
            predictions2[0]['boxes'].numpy()
        ]
        scores_list = [
            predictions1[0]['scores'].numpy(),
            predictions2[0]['scores'].numpy()
        ]
        labels_list = [
            predictions1[0]['labels'].numpy(),
            predictions2[0]['labels'].numpy()
        ]
        
        # Test WBF
        boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=[1.0, 1.0], iou_thr=0.55, skip_box_thr=0.0
        )
        
        self.assertIsInstance(boxes_wbf, np.ndarray)
        self.assertIsInstance(scores_wbf, np.ndarray)
        self.assertIsInstance(labels_wbf, np.ndarray)
        
        # Test standard NMS
        boxes_nms, scores_nms, labels_nms = nms(
            boxes_list, scores_list, labels_list,
            iou_thr=0.5, weights=[1.0, 1.0]
        )
        
        self.assertIsInstance(boxes_nms, np.ndarray)
        self.assertIsInstance(scores_nms, np.ndarray)
        self.assertIsInstance(labels_nms, np.ndarray)
    
    def test_real_yolo_performance(self):
        """Test performance characteristics of real YOLO integration"""
        import time
        
        # Measure inference time
        start_time = time.time()
        predictions = self.yolo_wrapper(self.test_image_tensor)
        inference_time = time.time() - start_time
        
        # Basic performance check (should complete in reasonable time)
        self.assertLess(inference_time, 10.0)  # Should complete within 10 seconds
        
        # Check memory usage (should not cause excessive memory usage)
        if len(predictions[0]['boxes']) > 0:
            # Verify tensor shapes are reasonable
            boxes = predictions[0]['boxes']
            self.assertEqual(boxes.shape[1], 4)  # 4 coordinates per box
            
            scores = predictions[0]['scores']
            self.assertEqual(scores.shape[0], boxes.shape[0])  # Same number of scores as boxes


class TestRealYOLOErrorHandling(unittest.TestCase):
    """Test error handling with real YOLO models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        try:
            from ultralytics import YOLO
            cls.yolo_model = YOLO('yolov8n.pt')
            cls.yolo_available = True
        except Exception:
            cls.yolo_available = False
    
    def setUp(self):
        """Set up for each test"""
        if not self.yolo_available:
            self.skipTest("YOLO model not available")
        
        self.yolo_wrapper = wrap_yolo(self.yolo_model, imsize=640)
    
    def test_yolo_invalid_image_format(self):
        """Test handling of invalid image formats"""
        # Test with invalid tensor shape
        invalid_tensor = torch.randn(3, 200, 200)  # Missing batch dimension
        
        with self.assertRaises((IndexError, ValueError, RuntimeError)):
            self.yolo_wrapper(invalid_tensor)
    
    def test_yolo_empty_image(self):
        """Test handling of empty images"""
        # Test with empty tensor
        empty_tensor = torch.empty(0, 3, 640, 640)
        
        with self.assertRaises((IndexError, ValueError, RuntimeError)):
            self.yolo_wrapper(empty_tensor)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 