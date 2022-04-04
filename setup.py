#!/usr/bin/env python

from setuptools import setup

setup(
    name='CompactionAnalyzer',
    packages=['CompactionAnalyzer'],
    version='1.0.1',
    description='A Python package to analyze tissue compaction around cells in fiber materials',
    url='',
    download_url = '',
    author='David Böhringer, Andreas Bauer',
    author_email='david.boehringer@fau.de',
    license='GNU General Public License v3.0',
    install_requires=['numpy>=1.16.2',
                      'pandas>=0.23.4',
                      'matplotlib>=2.2.2',
                      'roipoly>=0.5.2'
					  'pyyml',
					  'matplotlib-scalebar'],
    keywords = ['structure', 'contractility', 'compaction','fibrosis', 'biophysics'],
    classifiers = [],
    )
