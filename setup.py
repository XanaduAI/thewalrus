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
import sys
import os
import platform

from setuptools import find_packages


with open("thewalrus/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


requirements = [
    "numpy",
    "scipy>=1.2.1",
    "numba>=0.49.1",
    "dask[delayed]",
    "sympy>=1.5.1",
    "repoze.lru>=0.7",
]


setup_requirements = ["numpy", "pkgconfig"]


BUILD_EXT = True

try:
    import numpy as np
    from numpy.distutils.core import setup
    from numpy.distutils.extension import Extension
except ImportError:
    raise ImportError(
        "ERROR: NumPy needs to be installed first. "
        "You can install it with pip:"
        "\n\npip install numpy"
    )

EXTENSIONS = []
if BUILD_EXT:

    import pkgconfig
    from Cython.Build import cythonize

    CFLAGS = os.environ.get("CFLAGS", "-Wall -O3")

    USE_OPENBLAS = bool(os.environ.get("USE_OPENBLAS"))
    USE_LAPACK = bool(os.environ.get("USE_LAPACK")) or USE_OPENBLAS
    USE_OPENMP = platform.system() != "Windows"
    EIGEN_INCLUDE_DIR = os.environ.get("EIGEN_INCLUDE_DIR", "")

    CONFIG = {
        "sources": ["thewalrus/libwalrus.pyx"],
        "depends": [
            "include/libwalrus.hpp",
            "include/eigenvalue_hafnian.hpp",
            "include/recursive_hafnian.hpp",
            "include/repeated_hafnian.hpp",
            "include/hafnian_approx.hpp",
            "include/torontonian.hpp",
            "include/permanent.hpp",
            "include/hermite_multidimensional.hpp",
            "include/stdafx.h",
            "include/fsum.hpp",
        ],
        "extra_compile_args": [*{"-fPIC", "-std=c++14", *CFLAGS.split(" ")}],
        "extra_link_args": [],
        "include_dirs": ['./include'],
        "language": "c++",
        "libraries": ["eigen3"],
        "library_dirs": [np.get_include()],
    }

    if platform.system() == "Windows":
        CONFIG["extra_compile_args"].extend(("-static",))
        CONFIG["extra_link_args"].extend(
            ("-static", "-static-libgfortran", "-static-libgcc")
        )
    elif platform.system() == "Darwin":
        CONFIG["extra_compile_args"].extend(
            ("-Xpreprocessor", "-fopenmp", "-mmacosx-version-min=10.9", "-shared")
        )
        CONFIG["extra_link_args"].extend(("-Xpreprocessor", "-fopenmp"))
        CONFIG["library_dirs"].append(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/"
            "XcodeDefault.xctoolchain/usr/include/c++/v1/"
        )
    else:
        CONFIG["extra_compile_args"].extend(("-fopenmp", "-shared"))
        CONFIG["extra_link_args"].extend(("-fopenmp",))

    if EIGEN_INCLUDE_DIR:
        CONFIG["library_dirs"].append(EIGEN_INCLUDE_DIR)

    if USE_OPENBLAS:
        CONFIG["libraries"].append("openblas")

    if USE_LAPACK:
        CONFIG["libraries"].append("lapacke")
        CONFIG["extra_compile_args"].append("-DLAPACKE=1")

    for k, v in pkgconfig.parse(" ".join(CONFIG["libraries"])).items():
        if k in CONFIG:
            CONFIG[k].extend(v)
        else:
            CONFIG[k] = v

    EXTENSIONS = cythonize(
        [Extension("libwalrus", **CONFIG)],
        compile_time_env={"_OPENMP": USE_OPENMP, "LAPACKE": USE_LAPACK})

info = {
    "name": "thewalrus",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "nicolas@xanadu.ai",
    "url": "https://github.com/XanaduAI/thewalrus",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "description": "Open source library for hafnian calculation",
    "long_description": open("README.rst").read(),
    "provides": ["thewalrus"],
    "install_requires": requirements,
    "setup_requires": setup_requirements,
    "ext_modules": EXTENSIONS,
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
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
