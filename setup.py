#!/usr/bin/env python
import os, sys
import shutil
import datetime

from setuptools import setup, find_packages
from setuptools.command.install import install

readme = open('README.md').read()

VERSION = '0.1.5'

# import subprocess
# commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True).decode('UTF-8').rstrip()
# VERSION += "_" + str(int(commit_hash, 16))[:8]
VERSION += "_" + datetime.datetime.now().strftime('%Y%m%d%H%M')[2:]
# print(VERSION)

setup(
    # Metadata
    name='odach',
    version=VERSION,
    author='Kentaro Yoshioka',
    author_email='meathouse47@gmail.com',
    url='https://github.com/kentaroy47/ODA-Object-Detection-ttA',
    description='ODAch is a test-time-augmentation tool for pytorch 2d object detectors.',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    #nstall_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
