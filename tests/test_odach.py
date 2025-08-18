#!/usr/bin/env python3
"""
Test file for the odach library
Tests all major components including augmentation classes, NMS, WBF, and TTA wrapper
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add the current directory to path to import odach
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from odach.oda import (
    Base, HorizontalFlip, VerticalFlip, Rotate90Left, Rotate90Right,
    Multiply, MultiScale, MultiScaleFlip, MultiScaleHFlip, TTACompose,
    nms_func, TTAWrapper, wrap_effdet
)
from odach.nms import nms, soft_nms, nms_method
from odach.wbf import weighted_boxes_fusion, bb_intersection_over_union


class TestAugmentationClasses(unittest.TestCase):
    """Test all augmentation classes"""
    
    def setUp(self):
        # Create test image tensor (batch_size=1, channels=3, height=120, width=100)
        self.test_image = torch.randn(1, 3, 120, 100)
        self.test_batch = torch.randn(2, 3, 120, 100)
        self.test_boxes = np.array([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.8, 0.8]])
    
    def test_base_class(self):
        """Test Base class raises NotImplementedError"""
        base = Base()
        with self.assertRaises(NotImplementedError):
            base.augment(self.test_image)
        with self.assertRaises(NotImplementedError):
            base.batch_augment(self.test_batch)
        with self.assertRaises(NotImplementedError):
            base.deaugment_boxes(self.test_boxes)
    
    def test_horizontal_flip(self):
        """Test HorizontalFlip augmentation"""
        hflip = HorizontalFlip()
        
        # Test single image
        augmented = hflip.augment(self.test_image)
        self.assertEqual(augmented.shape, self.test_image.shape)
        self.assertEqual(hflip.imsize, 100)  # width
        
        # Test batch
        augmented_batch = hflip.batch_augment(self.test_batch)
        self.assertEqual(augmented_batch.shape, self.test_batch.shape)
        
        # Test box deaugmentation
        deaugmented_boxes = hflip.deaugment_boxes(self.test_boxes.copy())
        self.assertEqual(deaugmented_boxes.shape, self.test_boxes.shape)
        
        # Check that x coordinates are flipped correctly
        expected_x1 = 100 - self.test_boxes[:, 2]  # x1 = width - x2
        expected_x2 = 100 - self.test_boxes[:, 0]  # x2 = width - x1
        np.testing.assert_array_almost_equal(deaugmented_boxes[:, 0], expected_x1)
        np.testing.assert_array_almost_equal(deaugmented_boxes[:, 2], expected_x2)
    
    def test_vertical_flip(self):
        """Test VerticalFlip augmentation"""
        vflip = VerticalFlip()
        
        # Test single image
        augmented = vflip.augment(self.test_image)
        self.assertEqual(augmented.shape, self.test_image.shape)
        self.assertEqual(vflip.imsize, 120)  # height
        
        # Test batch
        augmented_batch = vflip.batch_augment(self.test_batch)
        self.assertEqual(augmented_batch.shape, self.test_batch.shape)
        
        # Test box deaugmentation
        deaugmented_boxes = vflip.deaugment_boxes(self.test_boxes.copy())
        self.assertEqual(deaugmented_boxes.shape, self.test_boxes.shape)
        
        # Check that y coordinates are flipped correctly
        expected_y1 = 120 - self.test_boxes[:, 3]  # y1 = height - y2
        expected_y2 = 120 - self.test_boxes[:, 1]  # y2 = height - y1
        np.testing.assert_array_almost_equal(deaugmented_boxes[:, 1], expected_y1)
        np.testing.assert_array_almost_equal(deaugmented_boxes[:, 3], expected_y2)
    
    def test_rotate90_left(self):
        """Test Rotate90Left augmentation"""
        rot_left = Rotate90Left()
        
        # Test single image
        augmented = rot_left.augment(self.test_image)
        # After 90-degree left rotation: [1, 3, 120, 100] -> [1, 3, 100, 120]
        expected_shape = (1, 3, 100, 120)
        self.assertEqual(augmented.shape, expected_shape)
        self.assertEqual(rot_left.imsize, 120)  # height
        
        # Test batch
        augmented_batch = rot_left.batch_augment(self.test_batch)
        expected_batch_shape = (2, 3, 100, 120)
        self.assertEqual(augmented_batch.shape, expected_batch_shape)
        
        # Test box deaugmentation
        deaugmented_boxes = rot_left.deaugment_boxes(self.test_boxes.copy())
        self.assertEqual(deaugmented_boxes.shape, self.test_boxes.shape)
    
    def test_rotate90_right(self):
        """Test Rotate90Right augmentation"""
        rot_right = Rotate90Right()
        
        # Test single image
        augmented = rot_right.augment(self.test_image)
        # After 90-degree right rotation: [1, 3, 120, 100] -> [1, 3, 100, 120]
        expected_shape = (1, 3, 100, 120)
        self.assertEqual(augmented.shape, expected_shape)
        self.assertEqual(rot_right.imsize, 120)  # height
        
        # Test batch
        augmented_batch = rot_right.batch_augment(self.test_batch)
        expected_batch_shape = (2, 3, 100, 120)
        self.assertEqual(augmented_batch.shape, expected_batch_shape)
        
        # Test box deaugmentation
        deaugmented_boxes = rot_right.deaugment_boxes(self.test_boxes.copy())
        self.assertEqual(deaugmented_boxes.shape, self.test_boxes.shape)
    
    def test_multiply(self):
        """Test Multiply augmentation"""
        scale = 1.5
        mult = Multiply(scale)
        
        # Test single image
        augmented = mult.augment(self.test_image)
        self.assertEqual(augmented.shape, self.test_image.shape)
        np.testing.assert_array_almost_equal(augmented.numpy(), self.test_image.numpy() * scale)
        
        # Test batch
        augmented_batch = mult.batch_augment(self.test_batch)
        self.assertEqual(augmented_batch.shape, self.test_batch.shape)
        np.testing.assert_array_almost_equal(augmented_batch.numpy(), self.test_batch.numpy() * scale)
        
        # Test box deaugmentation (should return unchanged)
        deaugmented_boxes = mult.deaugment_boxes(self.test_boxes.copy())
        np.testing.assert_array_almost_equal(deaugmented_boxes, self.test_boxes)
    
    def test_multiscale(self):
        """Test MultiScale augmentation"""
        scale = 1.2
        multiscale = MultiScale(scale)
        
        # Test single image
        augmented = multiscale.augment(self.test_image)
        expected_height = int(120 * scale)  # height: 120 * 1.2 = 144
        expected_width = int(100 * scale)   # width: 100 * 1.2 = 120
        self.assertEqual(augmented.shape, (1, 3, expected_height, expected_width))
        
        # Test batch
        augmented_batch = multiscale.batch_augment(self.test_batch)
        self.assertEqual(augmented_batch.shape, (2, 3, expected_height, expected_width))
        
        # Test box deaugmentation
        deaugmented_boxes = multiscale.deaugment_boxes(self.test_boxes.copy())
        np.testing.assert_array_almost_equal(deaugmented_boxes, self.test_boxes / scale)
    
    def test_tta_compose(self):
        """Test TTACompose class"""
        transforms = [HorizontalFlip(), Multiply(1.2)]
        compose = TTACompose(transforms)
        
        # Test single image
        augmented = compose.augment(self.test_image)
        self.assertEqual(augmented.shape, self.test_image.shape)
        
        # Test batch
        augmented_batch = compose.batch_augment(self.test_batch)
        self.assertEqual(augmented_batch.shape, self.test_batch.shape)
        
        # Test box deaugmentation
        deaugmented_boxes = compose.deaugment_boxes(self.test_boxes.copy())
        self.assertEqual(deaugmented_boxes.shape, self.test_boxes.shape)
        
        # Test prepare_boxes
        prepared_boxes = compose.prepare_boxes(self.test_boxes.copy())
        self.assertEqual(prepared_boxes.shape, self.test_boxes.shape)


class TestNMSFunctions(unittest.TestCase):
    """Test NMS and related functions"""
    
    def setUp(self):
        # Create test data
        self.boxes = [
            np.array([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]]),  # Model 1
            np.array([[0.15, 0.15, 0.35, 0.35], [0.6, 0.6, 0.8, 0.8]])  # Model 2
        ]
        self.scores = [
            np.array([0.9, 0.8]),  # Model 1
            np.array([0.85, 0.7])   # Model 2
        ]
        self.labels = [
            np.array([1, 1]),  # Model 1
            np.array([1, 2])   # Model 2
        ]
    
    def test_nms(self):
        """Test standard NMS function"""
        boxes, scores, labels = nms(self.boxes, self.scores, self.labels, iou_thr=0.5)
        
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(boxes), len(scores))
        self.assertEqual(len(boxes), len(labels))
    
    def test_soft_nms(self):
        """Test soft NMS function"""
        boxes, scores, labels = soft_nms(
            self.boxes, self.scores, self.labels, 
            method=2, iou_thr=0.5, sigma=0.5, thresh=0.001
        )
        
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(boxes), len(scores))
        self.assertEqual(len(boxes), len(labels))
    
    def test_nms_method(self):
        """Test NMS method function"""
        boxes, scores, labels = nms_method(
            self.boxes, self.scores, self.labels, 
            method=3, iou_thr=0.5, weights=[1.0, 1.0]
        )
        
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(boxes), len(scores))
        self.assertEqual(len(boxes), len(labels))


class TestWBFFunctions(unittest.TestCase):
    """Test Weighted Boxes Fusion functions"""
    
    def setUp(self):
        # Create test data
        self.boxes = [
            np.array([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]]),  # Model 1
            np.array([[0.15, 0.15, 0.35, 0.35], [0.6, 0.6, 0.8, 0.8]])  # Model 2
        ]
        self.scores = [
            np.array([0.9, 0.8]),  # Model 1
            np.array([0.85, 0.7])   # Model 2
        ]
        self.labels = [
            np.array([1, 1]),  # Model 1
            np.array([1, 2])   # Model 2
        ]
    
    def test_bb_intersection_over_union(self):
        """Test IoU calculation"""
        box1 = [0.1, 0.1, 0.3, 0.3]
        box2 = [0.2, 0.2, 0.4, 0.4]
        
        iou = bb_intersection_over_union(box1, box2)
        self.assertIsInstance(iou, float)
        self.assertGreaterEqual(iou, 0.0)
        self.assertLessEqual(iou, 1.0)
        
        # Test non-overlapping boxes
        box3 = [0.6, 0.6, 0.8, 0.8]
        iou_no_overlap = bb_intersection_over_union(box1, box3)
        self.assertEqual(iou_no_overlap, 0.0)
    
    def test_weighted_boxes_fusion(self):
        """Test WBF function"""
        boxes, scores, labels = weighted_boxes_fusion(
            self.boxes, self.scores, self.labels, 
            weights=[1.0, 1.0], iou_thr=0.55, skip_box_thr=0.0
        )
        
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(boxes), len(scores))
        self.assertEqual(len(boxes), len(labels))
        
        # Test with different confidence types
        boxes_avg, scores_avg, labels_avg = weighted_boxes_fusion(
            self.boxes, self.scores, self.labels, 
            weights=[1.0, 1.0], iou_thr=0.55, skip_box_thr=0.0, conf_type='avg'
        )
        
        boxes_max, scores_max, labels_max = weighted_boxes_fusion(
            self.boxes, self.scores, self.labels, 
            weights=[1.0, 1.0], iou_thr=0.55, skip_box_thr=0.0, conf_type='max'
        )
        
        self.assertEqual(len(boxes_avg), len(scores_avg))
        self.assertEqual(len(boxes_max), len(scores_max))


class TestNMSFuncClass(unittest.TestCase):
    """Test nms_func class"""
    
    def setUp(self):
        self.boxes = [
            np.array([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]]),
            np.array([[0.15, 0.15, 0.35, 0.35], [0.6, 0.6, 0.8, 0.8]])
        ]
        self.scores = [
            np.array([0.9, 0.8]),
            np.array([0.85, 0.7])
        ]
        self.labels = [
            np.array([1, 1]),
            np.array([1, 2])
        ]
    
    def test_wbf_nms(self):
        """Test WBF NMS"""
        nms_func_instance = nms_func(nmsname="wbf", weights=None, iou_thr=0.5, skip_box_thr=0.1)
        boxes, scores, labels = nms_func_instance(self.boxes, self.scores, self.labels)
        
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
    
    def test_standard_nms(self):
        """Test standard NMS"""
        nms_func_instance = nms_func(nmsname="nms", weights=None, iou_thr=0.5, skip_box_thr=0.1)
        boxes, scores, labels = nms_func_instance(self.boxes, self.scores, self.labels)
        
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
    
    def test_invalid_nms(self):
        """Test invalid NMS name raises error"""
        nms_func_instance = nms_func(nmsname="invalid", weights=None, iou_thr=0.5, skip_box_thr=0.1)
        with self.assertRaises(NotImplementedError):
            nms_func_instance(self.boxes, self.scores, self.labels)


class TestTTAWrapper(unittest.TestCase):
    """Test TTAWrapper class"""
    
    def setUp(self):
        # Create a mock model that returns expected format
        class MockModel:
            def __call__(self, img):
                batch_size = img.shape[0]
                results = []
                for i in range(batch_size):
                    results.append({
                        'boxes': torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.8, 0.8]]),
                        'scores': torch.tensor([0.9, 0.8]),
                        'labels': torch.tensor([1, 1])
                    })
                return results
        
        self.mock_model = MockModel()
        self.test_image = torch.randn(2, 3, 100, 100)
    
    def test_tta_wrapper_initialization(self):
        """Test TTAWrapper initialization"""
        tta = [HorizontalFlip, VerticalFlip]
        wrapper = TTAWrapper(
            model=self.mock_model, 
            tta=tta, 
            scale=[1], 
            nms="wbf", 
            iou_thr=0.5, 
            skip_box_thr=0.5
        )
        
        self.assertIsInstance(wrapper.ttas, list)
        self.assertEqual(wrapper.model, self.mock_model)
        self.assertIsInstance(wrapper.nms, nms_func)
    
    def test_tta_generation_monoscale(self):
        """Test TTA generation for monoscale"""
        tta = [HorizontalFlip, VerticalFlip]
        wrapper = TTAWrapper(
            model=self.mock_model, 
            tta=tta, 
            scale=[1], 
            nms="wbf"
        )
        
        # Should generate 4 TTA combinations (2^2)
        self.assertEqual(wrapper.tta_num(), 4)
    
    def test_tta_generation_multiscale(self):
        """Test TTA generation for multiscale"""
        tta = [HorizontalFlip, VerticalFlip]
        wrapper = TTAWrapper(
            model=self.mock_model, 
            tta=tta, 
            scale=[0.8, 1.0, 1.2], 
            nms="wbf"
        )
        
        # Should generate 4 * 3 = 12 TTA combinations
        self.assertEqual(wrapper.tta_num(), 12)


class TestWrapEffdet(unittest.TestCase):
    """Test wrap_effdet class"""
    
    def setUp(self):
        # Create a mock EfficientDet model
        class MockEffDet:
            def __call__(self, img, img_info):
                batch_size = img.shape[0]
                # Return format: [batch_size, num_detections, 5] (x1, y1, x2, y2, score)
                detections = torch.zeros(batch_size, 2, 5)
                detections[:, 0, :4] = torch.tensor([0.1, 0.1, 0.3, 0.3])  # x1, y1, x2, y2
                detections[:, 0, 4] = 0.9  # score
                detections[:, 1, :4] = torch.tensor([0.6, 0.6, 0.8, 0.8])
                detections[:, 1, 4] = 0.8
                return detections
        
        self.mock_effdet = MockEffDet()
        self.test_images = torch.randn(2, 3, 512, 512)
    
    def test_wrap_effdet_initialization(self):
        """Test wrap_effdet initialization"""
        wrapper = wrap_effdet(self.mock_effdet, imsize=512)
        
        self.assertEqual(wrapper.model, self.mock_effdet)
        self.assertEqual(wrapper.imsize, 512)
    
    def test_wrap_effdet_call(self):
        """Test wrap_effdet call method"""
        wrapper = wrap_effdet(self.mock_effdet, imsize=512)
        
        # Note: This will fail in test environment due to CUDA dependency
        # In real usage, the model would be on CUDA
        try:
            predictions = wrapper(self.test_images, score_threshold=0.22)
            self.assertIsInstance(predictions, list)
            self.assertEqual(len(predictions), 2)
            
            for pred in predictions:
                self.assertIn('boxes', pred)
                self.assertIn('scores', pred)
                self.assertIn('labels', pred)
        except RuntimeError as e:
            if "CUDA" in str(e):
                # Skip CUDA-related errors in test environment
                pass
            else:
                raise


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire library"""
    
    def test_full_pipeline(self):
        """Test a complete TTA pipeline"""
        # Create test data
        test_image = torch.randn(1, 3, 100, 100)
        test_boxes = np.array([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.8, 0.8]])
        
        # Test horizontal flip + multiply
        transforms = [HorizontalFlip(), Multiply(1.2)]
        compose = TTACompose(transforms)
        
        # Apply augmentation
        augmented = compose.augment(test_image)
        self.assertEqual(augmented.shape, test_image.shape)
        
        # Deaugment boxes
        deaugmented_boxes = compose.deaugment_boxes(test_boxes.copy())
        self.assertEqual(deaugmented_boxes.shape, test_boxes.shape)
        
        # Test that boxes are properly transformed back
        # (This is a basic check - in real usage you'd verify specific transformations)
        self.assertTrue(np.all(deaugmented_boxes >= 0))
        self.assertTrue(np.all(deaugmented_boxes <= 100))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 