#!/usr/bin/env python

from setuptools import setup

setup(
    name='CompactionAnalyzer',
    packages=['CompactionAnalyzer'],
    version='1.0',
    description='A Python package to analyze tissue compaction around cells in fiber materials',
    url='',
    download_url = '',
    author='David Böhringer, Andreas Bauer',
    author_email='david.boehringer@fau.de',
    license='The MIT License (MIT)',
    install_requires=['numpy>=1.16.2',
                      'pandas>=0.23.4',
                      'matplotlib>=2.2.2',
                      'roipoly>=0.5.2'],
    keywords = ['structure', 'contractility', 'compaction','fibrosis', 'biophysics'],
    classifiers = [],
    )
