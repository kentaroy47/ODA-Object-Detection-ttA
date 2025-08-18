# Testing the ODACH Library

This directory contains comprehensive tests for the `odach` library, which provides test-time augmentation (TTA) functionality for PyTorch 2D object detectors.

## Test Files

### `test_odach.py`
The main test file containing comprehensive tests for all components of the odach library:

- **TestAugmentationClasses**: Tests all augmentation classes (HorizontalFlip, VerticalFlip, Rotate90Left, Rotate90Right, Multiply, MultiScale, etc.)
- **TestNMSFunctions**: Tests NMS (Non-Maximum Suppression) functions including standard NMS and soft NMS
- **TestWBFFunctions**: Tests Weighted Boxes Fusion (WBF) functions
- **TestNMSFuncClass**: Tests the nms_func wrapper class
- **TestTTAWrapper**: Tests the main TTAWrapper class
- **TestWrapEffdet**: Tests the EfficientDet wrapper
- **TestIntegration**: Integration tests for the complete pipeline

### `run_tests.py`
A simple test runner script that can execute all tests or specific test classes.

### `requirements.txt`
Lists the required dependencies for running the tests.

## Running the Tests

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
python test_odach.py
```

### Run Tests with Test Runner
```bash
# Run all tests
python run_tests.py

# Run specific test class
python run_tests.py TestAugmentationClasses
python run_tests.py TestNMSFunctions
python run_tests.py TestWBFFunctions
```

### Run with unittest directly
```bash
python -m unittest test_odach.py -v
```

## What the Tests Cover

### Augmentation Classes
- **Base**: Abstract base class that raises NotImplementedError
- **HorizontalFlip**: Horizontal image flipping with proper box coordinate transformation
- **VerticalFlip**: Vertical image flipping with proper box coordinate transformation
- **Rotate90Left/Right**: 90-degree rotations with box coordinate adjustments
- **Multiply**: Brightness adjustment (no box changes)
- **MultiScale**: Image scaling with box coordinate scaling
- **MultiScaleFlip**: Combined scaling and flipping
- **TTACompose**: Composition of multiple augmentations

### NMS Functions
- Standard NMS implementation
- Soft NMS with linear and Gaussian methods
- Box coordinate validation and correction
- Multi-label NMS support

### Weighted Boxes Fusion (WBF)
- IoU calculation between bounding boxes
- Box filtering and clustering
- Confidence score aggregation
- Support for different confidence types (avg/max)

### TTA Wrapper
- TTA generation for monoscale and multiscale scenarios
- Model inference integration
- Box deaugmentation pipeline
- NMS integration

### EfficientDet Wrapper
- Model output format conversion
- Score thresholding
- Box coordinate normalization

## Test Data
The tests use synthetic data:
- Random image tensors (batch_size × channels × height × width)
- Synthetic bounding boxes with normalized coordinates [0, 1]
- Mock model outputs that match expected formats

## Notes
- Some tests may fail in environments without CUDA (GPU) support
- The tests are designed to be run in the same directory as the `odach` package
- Mock models are used to avoid requiring actual trained models

## Expected Output
When running the tests, you should see output similar to:
```
test_base_class (__main__.TestAugmentationClasses) ... ok
test_horizontal_flip (__main__.TestAugmentationClasses) ... ok
test_vertical_flip (__main__.TestAugmentationClasses) ... ok
...
----------------------------------------------------------------------
Ran X tests in Xs

OK
```

## Troubleshooting
- **Import errors**: Make sure you're in the correct directory and the `odach` package is accessible
- **CUDA errors**: Some tests may fail without GPU support - this is expected
- **Dependency issues**: Install all required packages from `requirements.txt` 