#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='biogeochem_tools',
      version='24.6.0-alpha-1',

      description='Tools for processing of Coastal Environmental Baseline Program physical and biogeochemical data',
      long_description=readme(),

      classifiers=[
        'Development Status :: 3 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Oceanography',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
      ],

      license='MIT',

      packages=['biogeochem_tools'],
      
      install_requires=[
        'numpy',
        'pandas',
        'xarray==2022.11.0',
        'gsw',
        'pyrsktools',
        'pyco2sys',
        'calkulate'
      ],
      
      include_package_data=True,
      zip_safe=False
     )
