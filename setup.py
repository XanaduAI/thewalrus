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
from distutils.command.build_ext import build_ext


class build_ext(build_ext):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypes)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)


class CTypes(Extension):
    pass


with open("hafnian/_version.py") as f:
	version = f.readlines()[-1].split()[-1].strip("\"'")

# cmdclass = {'build_docs': BuildDoc}

cflags_default = "-std=c99 -O3 -Wall -fPIC -shared -fopenmp"

LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', "").split(":")
C_INCLUDE_PATH = os.environ.get('C_INCLUDE_PATH', "").split(":")
CFLAGS = os.environ.get('CFLAGS', cflags_default).split()

requirements = [
    "numpy>=1.13"
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
                    'hafnian'
                ],
    # 'package_data': {'hafnian': ['backends/data/*']},
    # 'include_package_data': True,
    'description': 'Open source library for hafnian calculation',
    'long_description': open('README.rst').read(),
    'provides': ["hafnian"],
    'install_requires': requirements,
    # 'extras_require': extra_requirements,
    'ext_package': 'hafnian.lib',
    'ext_modules': [
        CTypes("lhafnian",
            sources=["src/lhafnian.c",],
            depends=["src/lhafnian.h"],
            include_dirs=['/usr/local/include', '/usr/include', './src'] + C_INCLUDE_PATH,
            libraries=['lapacke'],
            library_dirs=['/usr/lib', '/usr/local/lib'] + LD_LIBRARY_PATH,
            extra_compile_args=CFLAGS,
            extra_link_args=['-fopenmp']
            ),
        CTypes("rlhafnian",
            sources=["src/rlhafnian.c"],
            depends=["src/rlhafnian.h"],
            include_dirs=['/usr/local/include', '/usr/include', './src'] + C_INCLUDE_PATH,
            libraries=['lapacke'],
            library_dirs=['/usr/lib', '/usr/local/lib'] + LD_LIBRARY_PATH,
            extra_compile_args=CFLAGS,
            extra_link_args=['-fopenmp']
            ),
        CTypes("hlhafnian",
            sources=["src/hlhafnian.c"],
            depends=["src/hlhafnian.h"],
            include_dirs=['/usr/local/include', '/usr/include', './src'] + C_INCLUDE_PATH,
            libraries=['lapacke'],
            library_dirs=['/usr/lib', '/usr/local/lib'] + LD_LIBRARY_PATH,
            extra_compile_args=CFLAGS,
            extra_link_args=['-fopenmp']
            )
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
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))
