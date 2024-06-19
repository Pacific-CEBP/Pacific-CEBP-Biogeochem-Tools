#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='biogeochem',
      version='24.6.0-beta',

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
        'numpy==1.26.4',
        'pandas==1.5.3',
        'xarray==2022.11.0',
        'gsw==3.6.18',
        'pyrsktools==1.1.1',
        'pyco2sys==1.8.3.1',
        'calkulate==23.6.1'
      ],
      
      include_package_data=True,
      zip_safe=False
     )
