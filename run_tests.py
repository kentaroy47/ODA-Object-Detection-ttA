#!/usr/bin/env python3
"""
Simple test runner for the odach library tests
"""

import sys
import os
import unittest

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    """Run all tests"""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

def run_specific_test(test_class_name):
    """Run a specific test class"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'test_odach.{test_class_name}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test class
        test_class = sys.argv[1]
        exit_code = run_specific_test(test_class)
    else:
        # Run all tests
        exit_code = run_tests()
    
    sys.exit(exit_code) 