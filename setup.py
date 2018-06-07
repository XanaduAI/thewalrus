# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
import sys
import os
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np


with open("hafnian/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

# cmdclass = {'build_docs': BuildDoc}

if os.name == 'nt':
    cflags_default = "-std=c99 -static -O3 -Wall -fPIC -shared -fopenmp -lopenblas"
    libraries = ['openblas']
    extra_link_args = ['-fopenmp', '-lopenblas']
else:
    cflags_default = "-std=c99 -O3 -Wall -fPIC -shared -fopenmp -llapacke"
    libraries = ['lapacke']
    extra_link_args = ['-fopenmp', '-llapacke']

LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', "").split(":")
C_INCLUDE_PATH = os.environ.get('C_INCLUDE_PATH', "").split(":") + [np.get_include()]
CFLAGS = os.environ.get('CFLAGS', cflags_default).split() + ['-I{}'.format(np.get_include())]

requirements = [
    "numpy>=1.13",
    "cython"
]

os.environ['OPT'] = ''

info = {
    'name': 'Hafnian',
    'version': version,
    'maintainer': 'Xanadu Inc.',
    'maintainer_email': 'nicolas@xanadu.ai',
    'url': 'http://xanadu.ai',
    'license': 'Apache License 2.0',
    'packages': [
                    'hafnian',
                    'hafnian.tests'
                ],
    'description': 'Open source library for hafnian calculation',
    'long_description': open('README.rst').read(),
    'provides': ["hafnian"],
    'install_requires': requirements,
    'ext_package': 'hafnian.lib',
    'ext_modules': [
        Extension("libhaf",
            sources=["hafnian/lhafnian.pyx", "src/lhafnian.c",],
            depends=["src/lhafnian.h"],
            include_dirs=C_INCLUDE_PATH,
            libraries=libraries,
            library_dirs=['/usr/lib', '/usr/local/lib'] + LD_LIBRARY_PATH,
            extra_compile_args=CFLAGS,
            extra_link_args=extra_link_args),
        Extension("librhaf",
            sources=["hafnian/rlhafnian.pyx", "src/rlhafnian.c"],
            depends=["src/rlhafnian.h"],
            include_dirs=C_INCLUDE_PATH,
            libraries=libraries,
            library_dirs=['/usr/lib', '/usr/local/lib'] + LD_LIBRARY_PATH,
            extra_compile_args=CFLAGS,
            extra_link_args=extra_link_args)
    ],
    'cmdclass': {'build_ext': build_ext},
    'command_options': {
        'build_sphinx': {
            'version': ('setup.py', version),
            'release': ('setup.py', version)}}
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))
