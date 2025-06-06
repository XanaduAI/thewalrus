# Copyright 2019-2022 Xanadu Quantum Technologies Inc.

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
from setuptools import find_packages, setup


def get_version():
    with open("thewalrus/_version.py") as f:
        return f.readlines()[-1].split()[-1].strip("\"'")


info = {
    "name": "thewalrus",
    "version": get_version(),
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/XanaduAI/thewalrus",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "description": "Open source library for hafnian calculation",
    "long_description": open("README.rst").read(),
    "provides": ["thewalrus"],
    "install_requires": [
        "dask[delayed]",
        "numba>=0.61.2,<1",
        "numpy>=2.0.0,<3",
        "scipy>=1.15.3,<2",
        "sympy>=1.14.0,<2",
    ],
    "setup_requires": ["numpy"],
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
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
