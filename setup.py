#!/usr/bin/env python
from pathlib import Path
from setuptools import setup

long_description = (Path(__file__).parent / "README_PYPI.md").read_text(encoding="utf-8")

setup(
    name='CompactionAnalyzer',
    packages=['CompactionAnalyzer'],
    version='1.0.9',
    description='Structure tensor-based image orientation analysis with a Python API, Saenopy GUI and browser tools.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/davidbhr/CompactionAnalyzer',
    project_urls={
        'Documentation': 'https://github.com/davidbhr/CompactionAnalyzer#readme',
        'Source': 'https://github.com/davidbhr/CompactionAnalyzer',
        'Issues': 'https://github.com/davidbhr/CompactionAnalyzer/issues',
        'Python API': 'https://github.com/davidbhr/CompactionAnalyzer/blob/master/docs/direct_api.md',
        'Examples': 'https://github.com/davidbhr/CompactionAnalyzer#examples-by-analysis-mode',
        'Browser app': 'https://davidbhr.github.io/CompactionAnalyzer/',
        'Saenopy GUI': 'https://github.com/rgerum/saenopy/releases',
    },
    author='David Böhringer, Andreas Bauer',
    author_email='david.boehringer@fau.de',
    license='GNU General Public License v3.0',
    install_requires=['numpy>=1.2.6',
                      'pandas>=0.23.4',
                      'matplotlib>=2.2.2',
                      'roipoly>=0.5.3',
					  'scikit-image>=0.24.0',			  
					  'PyYAML',
					  'tqdm',
					  'openpyxl',
					  'natsort',
					  'matplotlib-scalebar'],
    keywords=[
        'structure tensor', 'image orientation', 'fiber orientation', 'fibre orientation',
        'fiber alignment', 'angular deviation', 'coherency', 'coherence', 'anisotropy',
        'microscopy', 'collagen', 'actin', 'cell contractility', 'compaction',
        'fibrosis', 'biophysics', 'Python API', 'GUI',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    )
