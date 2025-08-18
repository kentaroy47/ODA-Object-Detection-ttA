#!/usr/bin/env python3
"""
Test file for YOLO-specific functionality in the odach library
Tests the wrap_yolo class and TTA integration with YOLO models
"""

import unittest
import numpy as np
import torch
import sys
import os
from unittest.mock import Mock

# Add the current directory to path to import odach
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from odach.oda import (
    wrap_yolo, TTAWrapper, HorizontalFlip, VerticalFlip, 
    Rotate90Left, Multiply
)
from odach.nms import nms
from odach.wbf import weighted_boxes_fusion


class MockYOLOResults:
    """Mock YOLO results object to simulate different YOLO versions"""
    
    def __init__(self, boxes, scores, labels, version="v8"):
        self.version = version
        if version == "v8":
            # YOLOv8+ format
            self.boxes = Mock()
            self.boxes.xyxy = torch.tensor(boxes)
            self.boxes.conf = torch.tensor(scores)
            self.boxes.cls = torch.tensor(labels)
        elif version == "v5":
            # YOLOv5 format
            self.xyxy = [torch.tensor(np.column_stack([boxes, scores, labels]))]
        else:
            # Generic format
            self.boxes = None
            self.xyxy = None


class TestWrapYOLO(unittest.TestCase):
    """Test the wrap_yolo class for YOLO model integration"""
    
    def setUp(self):
        # Create mock YOLO model
        self.mock_yolo_model = Mock()
        
        # Test data
        self.test_boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]])
        self.test_scores = np.array([0.9, 0.8])
        self.test_labels = np.array([0, 1])
        
        # Test image tensor (batch_size=2, channels=3, height=200, width=200)
        self.test_image = torch.randn(2, 3, 200, 200)
        
        # Create wrapper instances
        self.yolo_wrapper_v8 = wrap_yolo(self.mock_yolo_model, imsize=640, score_threshold=0.25)
        self.yolo_wrapper_v5 = wrap_yolo(self.mock_yolo_model, imsize=640, score_threshold=0.25)
    
    def test_wrap_yolo_initialization(self):
        """Test wrap_yolo initialization"""
        wrapper = wrap_yolo(self.mock_yolo_model)
        
        self.assertEqual(wrapper.model, self.mock_yolo_model)
        self.assertEqual(wrapper.imsize, 640)
        self.assertEqual(wrapper.score_threshold, 0.25)
        self.assertEqual(wrapper.iou_threshold, 0.45)
        
        # Test custom parameters
        wrapper_custom = wrap_yolo(self.mock_yolo_model, imsize=1024, score_threshold=0.5, iou_threshold=0.3)
        self.assertEqual(wrapper_custom.imsize, 1024)
        self.assertEqual(wrapper_custom.score_threshold, 0.5)
        self.assertEqual(wrapper_custom.iou_threshold, 0.3)
    
    def test_wrap_yolo_yolov8_format(self):
        """Test wrap_yolo with YOLOv8+ output format"""
        # Mock YOLOv8 results
        mock_results = MockYOLOResults(self.test_boxes, self.test_scores, self.test_labels, "v8")
        self.mock_yolo_model.return_value = mock_results
        
        # Run inference
        predictions = self.yolo_wrapper_v8(self.test_image)
        
        # Check output format
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)  # batch size
        
        for pred in predictions:
            self.assertIn('boxes', pred)
            self.assertIn('scores', pred)
            self.assertIn('labels', pred)
            self.assertIsInstance(pred['boxes'], torch.Tensor)
            self.assertIsInstance(pred['scores'], torch.Tensor)
            self.assertIsInstance(pred['labels'], torch.Tensor)
    
    def test_wrap_yolo_yolov5_format(self):
        """Test wrap_yolo with YOLOv5 output format"""
        # Mock YOLOv5 results
        mock_results = MockYOLOResults(self.test_boxes, self.test_scores, self.test_labels, "v5")
        self.mock_yolo_model.return_value = mock_results
        
        # Run inference
        predictions = self.yolo_wrapper_v5(self.test_image)
        
        # Check output format
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)  # batch size
        
        for pred in predictions:
            self.assertIn('boxes', pred)
            self.assertIn('scores', pred)
            self.assertIn('labels', pred)
    
    def test_wrap_yolo_score_filtering(self):
        """Test score threshold filtering in wrap_yolo"""
        # Create boxes with different scores
        low_score_boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]])
        low_scores = np.array([0.1, 0.2])  # Below threshold
        labels = np.array([0, 1])
        
        mock_results = MockYOLOResults(low_score_boxes, low_scores, labels, "v8")
        self.mock_yolo_model.return_value = mock_results
        
        # Run with high threshold
        predictions = self.yolo_wrapper_v8(self.test_image, score_threshold=0.5)
        
        # Should filter out low confidence detections
        for pred in predictions:
            self.assertEqual(len(pred['scores']), 0)
            self.assertEqual(len(pred['boxes']), 0)
            self.assertEqual(len(pred['labels']), 0)
    
    def test_wrap_yolo_box_normalization(self):
        """Test that boxes are properly normalized to [0, 1] range"""
        # Create boxes with absolute coordinates
        abs_boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]])
        scores = np.array([0.9, 0.8])
        labels = np.array([0, 1])
        
        mock_results = MockYOLOResults(abs_boxes, scores, labels, "v8")
        self.mock_yolo_model.return_value = mock_results
        
        # Run inference
        predictions = self.yolo_wrapper_v8(self.test_image)
        
        # Check that boxes are normalized
        for pred in predictions:
            if len(pred['boxes']) > 0:
                boxes = pred['boxes'].numpy()
                # All coordinates should be in [0, 1] range
                self.assertTrue(np.all(boxes >= 0))
                self.assertTrue(np.all(boxes <= 1))
    
    def test_wrap_yolo_empty_detections(self):
        """Test wrap_yolo with no detections"""
        # Mock empty results
        empty_boxes = np.empty((0, 4))
        empty_scores = np.empty(0)
        empty_labels = np.empty(0, dtype=int)
        
        mock_results = MockYOLOResults(empty_boxes, empty_scores, empty_labels, "v8")
        self.mock_yolo_model.return_value = mock_results
        
        # Run inference
        predictions = self.yolo_wrapper_v8(self.test_image)
        
        # Check output format for empty detections
        for pred in predictions:
            self.assertEqual(len(pred['boxes']), 0)
            self.assertEqual(len(pred['scores']), 0)
            self.assertEqual(len(pred['labels']), 0)


class TestYOLOTTAIntegration(unittest.TestCase):
    """Test TTA integration with YOLO models"""
    
    def setUp(self):
        # Create mock YOLO model
        self.mock_yolo_model = Mock()
        
        # Create YOLO wrapper
        self.yolo_wrapper = wrap_yolo(self.mock_yolo_model, imsize=640)
        
        # Test image
        self.test_image = torch.randn(1, 3, 200, 200)
        
        # Test detection results
        self.test_boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]])
        self.test_scores = np.array([0.9, 0.8])
        self.test_labels = np.array([0, 1])
    
    def test_yolo_tta_wrapper_creation(self):
        """Test creating TTAWrapper with YOLO wrapper"""
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
        
        self.assertIsInstance(tta_wrapper, TTAWrapper)
        self.assertEqual(tta_wrapper.model, self.yolo_wrapper)
        self.assertGreater(tta_wrapper.tta_num(), 0)
    
    def test_yolo_tta_multiscale(self):
        """Test YOLO TTA with multiple scales"""
        tta_transforms = [HorizontalFlip(), VerticalFlip()]
        scales = [0.8, 1.0, 1.2]
        
        # Create TTA wrapper with multiple scales
        tta_wrapper = TTAWrapper(
            model=self.yolo_wrapper,
            tta=tta_transforms,
            scale=scales,
            nms="wbf"
        )
        
        # Should generate more TTA combinations with scales
        expected_tta_count = len(scales) * (2 ** len(tta_transforms))
        self.assertEqual(tta_wrapper.tta_num(), expected_tta_count)
    
    def test_yolo_tta_augmentation_chain(self):
        """Test YOLO TTA with augmentation chain"""
        # Create complex TTA chain
        tta_transforms = [
            HorizontalFlip(),
            VerticalFlip(),
            Rotate90Left(),
            Multiply(1.1)
        ]
        
        tta_wrapper = TTAWrapper(
            model=self.yolo_wrapper,
            tta=tta_transforms,
            scale=[1],
            nms="wbf"
        )
        
        # Should generate 2^4 = 16 TTA combinations
        self.assertEqual(tta_wrapper.tta_num(), 16)


class TestYOLONMSIntegration(unittest.TestCase):
    """Test NMS integration with YOLO outputs"""
    
    def setUp(self):
        # Create test data
        self.boxes_list = [
            np.array([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]]),  # Model 1
            np.array([[0.15, 0.15, 0.35, 0.35], [0.6, 0.6, 0.8, 0.8]])  # Model 2
        ]
        self.scores_list = [
            np.array([0.9, 0.8]),  # Model 1
            np.array([0.85, 0.7])   # Model 2
        ]
        self.labels_list = [
            np.array([0, 0]),  # Model 1
            np.array([0, 1])   # Model 2
        ]
    
    def test_yolo_wbf_integration(self):
        """Test WBF integration with YOLO-style outputs"""
        # Test WBF with YOLO outputs
        boxes, scores, labels = weighted_boxes_fusion(
            self.boxes_list, self.scores_list, self.labels_list,
            weights=[1.0, 1.0], iou_thr=0.55, skip_box_thr=0.0
        )
        
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(boxes), len(scores))
        self.assertEqual(len(boxes), len(labels))
    
    def test_yolo_nms_integration(self):
        """Test NMS integration with YOLO-style outputs"""
        # Test standard NMS with YOLO outputs
        boxes, scores, labels = nms(
            self.boxes_list, self.scores_list, self.labels_list,
            iou_thr=0.5, weights=[1.0, 1.0]
        )
        
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(boxes), len(scores))
        self.assertEqual(len(boxes), len(labels))


class TestYOLORealWorldScenarios(unittest.TestCase):
    """Test YOLO integration with real-world scenarios"""
    
    def setUp(self):
        self.mock_yolo_model = Mock()
        self.yolo_wrapper = wrap_yolo(self.mock_yolo_model, imsize=640)
    
    def test_yolo_batch_processing(self):
        """Test YOLO wrapper with batch processing"""
        # Create batch of images with different sizes
        batch_images = [
            torch.randn(3, 640, 640),  # Square image
            torch.randn(3, 480, 640),  # Rectangular image
        ]
        
        # Mock YOLO results for each image
        results1 = MockYOLOResults(
            np.array([[10, 10, 50, 50]]),
            np.array([0.9]),
            np.array([0]),
            "v8"
        )
        results2 = MockYOLOResults(
            np.array([[100, 100, 150, 150]]),
            np.array([0.8]),
            np.array([1]),
            "v8"
        )
        
        self.mock_yolo_model.side_effect = [results1, results2]
        
        # Process batch
        predictions = self.yolo_wrapper(batch_images)
        
        # Check batch processing
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(predictions[0]['boxes']), 1)
        self.assertEqual(len(predictions[1]['boxes']), 1)
    
    def test_yolo_confidence_variations(self):
        """Test YOLO wrapper with different confidence levels"""
        # Test boxes with varying confidence scores
        boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150], [200, 200, 250, 250]])
        scores = np.array([0.95, 0.75, 0.45])  # High, medium, low confidence
        labels = np.array([0, 1, 2])
        
        mock_results = MockYOLOResults(boxes, scores, labels, "v8")
        self.mock_yolo_model.return_value = mock_results
        
        # Test with different thresholds
        predictions_low = self.yolo_wrapper(self.test_image, score_threshold=0.3)
        predictions_high = self.yolo_wrapper(self.test_image, score_threshold=0.8)
        
        # Low threshold should keep more detections
        self.assertGreaterEqual(len(predictions_low[0]['scores']), len(predictions_high[0]['scores']))


class TestYOLOErrorHandling(unittest.TestCase):
    """Test error handling in YOLO wrapper"""
    
    def setUp(self):
        self.mock_yolo_model = Mock()
        self.yolo_wrapper = wrap_yolo(self.mock_yolo_model, imsize=640)
        self.test_image = torch.randn(1, 3, 200, 200)
    
    def test_yolo_invalid_input(self):
        """Test YOLO wrapper with invalid input"""
        # Test with None input
        with self.assertRaises((TypeError, AttributeError)):
            self.yolo_wrapper(None)
        
        # Test with empty tensor
        empty_tensor = torch.empty(0, 3, 200, 200)
        with self.assertRaises((IndexError, ValueError)):
            self.yolo_wrapper(empty_tensor)
    
    def test_yolo_model_failure(self):
        """Test YOLO wrapper when model fails"""
        # Mock model to raise exception
        self.mock_yolo_model.side_effect = RuntimeError("Model inference failed")
        
        with self.assertRaises(RuntimeError):
            self.yolo_wrapper(self.test_image)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 