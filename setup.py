#!/usr/bin/env python
# -*- coding: utf-8 -*

import os

from setuptools import find_packages, setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

VERSION = '0.1'

setup(
    name='comp',
    version=VERSION,
    packages=find_packages(exclude=('comp/tests', )),
    include_package_data=True,
    description='Kaggle competition tools',
    install_requires=[
        'numpy==1.15.4',
        'GitPython==2.1.11',
        'scikit_learn==0.20.3'
    ],
)
