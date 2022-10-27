#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='biogeochem',
      version='1.0.0b1',

      description='Tools for processing of Coastal Environmental Baseline Program physical and biogeochemical data',
      long_description=readme(),

      classifiers=[
        'Development Status :: 3 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Oceanography',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
      ],

      license='MIT',

      packages=['biogeochem'],
      
      install_requires=[
        'numpy',
        'pandas',
        'xarray',
      ],
      
      include_package_data=True,
      zip_safe=False
     )
