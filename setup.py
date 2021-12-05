# Copyright 2019 Xanadu Quantum Technologies Inc.

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
import os
import platform

from setuptools import find_packages

try:
    import numpy as np
    from numpy.distutils.core import setup
    from numpy.distutils.extension import Extension
except ImportError as exc:
    raise ImportError(
        "Numpy must be installed to build The Walrus."
        "You can install it with pip:"
        "\n\npip install numpy"
    ) from exc


def get_version():
    with open("thewalrus/_version.py") as f:
        return f.readlines()[-1].split()[-1].strip("\"'")


info = {
    "name": "thewalrus",
    "version": get_version(),
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "nicolas@xanadu.ai",
    "url": "https://github.com/XanaduAI/thewalrus",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "description": "Open source library for hafnian calculation",
    "long_description": open("README.rst").read(),
    "provides": ["thewalrus"],
    "install_requires": [
        "dask[delayed]",
        "numba>=0.49.1,<0.54",
        "scipy>=1.2.1",
        "sympy>=1.5.1",
    ],
    "setup_requires": ["cython", "numpy"],
    "ext_package": "thewalrus",
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
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
