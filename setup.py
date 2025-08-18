#!/usr/bin/env python
from setuptools import setup, find_packages

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

VERSION = '0.3.0'

setup(
    # Metadata
    name='odach',
    version=VERSION,
    author='Kentaro Yoshioka',
    author_email='meathouse47@gmail.com',
    url='https://github.com/kentaroy47/ODA-Object-Detection-ttA',
    description='ODAch is a test-time-augmentation tool for pytorch 2d object detectors with YOLO support.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='object-detection, pytorch, tta, test-time-augmentation, yolo, computer-vision',

    # Package info
    packages=find_packages(exclude=('tests*', 'test_*', '*test*')),
    include_package_data=True,
    zip_safe=False,

    # Dependencies
    install_requires=requirements,
    python_requires='>=3.7',

    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/kentaroy47/ODA-Object-Detection-ttA/issues',
        'Source': 'https://github.com/kentaroy47/ODA-Object-Detection-ttA',
        'Documentation': 'https://github.com/kentaroy47/ODA-Object-Detection-ttA#readme',
    },
)
