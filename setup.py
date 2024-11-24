#!/usr/bin/env python
from setuptools import setup

setup(
    name='CompactionAnalyzer',
    packages=['CompactionAnalyzer'],
    version='1.0.4',
    description='A python package to analyze matrix fiber alignment around cells (proxy for cellular forces)',
    url='',
    download_url = '',
    author='David Böhringer, Andreas Bauer',
    author_email='david.boehringer@fau.de',
    license='GNU General Public License v3.0',
    install_requires=['numpy>=1.2.6',
                      'pandas>=0.23.4',
                      'matplotlib>=2.2.2',
                      'roipoly>=0.5.3',
					  'scikit-image>=0.24.0',			  
					  'pyyml',
					  'tqdm',
					  'openpyxl',
					  'natsort',
					  'matplotlib-scalebar'],
    keywords = ['structure', 'contractility', 'compaction','fibrosis', 'biophysics'],
    classifiers = [],
    )
