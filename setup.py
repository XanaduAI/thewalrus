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

import setuptools


with open("thewalrus/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


requirements = [
    "numpy",
    "scipy>=1.2.1",
    "numba>=0.43.1"
]


setup_requirements = [
    "numpy"
]


BUILD_EXT = True

try:
    import numpy as np
    from numpy.distutils.core import setup
    from numpy.distutils.extension import Extension
except ImportError:
    raise ImportError("ERROR: NumPy needs to be installed first. "
                      "You can install it with pip:"
                      "\n\npip install numpy")


if BUILD_EXT:

    USE_CYTHON = True
    try:
        from Cython.Build import cythonize
        ext = 'pyx'
    except:
        def cythonize(x, compile_time_env=None):
            return x

        USE_CYTHON = False
        cythonize = cythonize
        ext = 'cpp'


    library_default = ""
    USE_OPENMP = True
    EIGEN_INCLUDE = [os.environ.get("EIGEN_INCLUDE_DIR", ""), "/usr/local/include/eigen3", "/usr/include/eigen3"]

    LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', library_default).split(":")
    C_INCLUDE_PATH = os.environ.get('C_INCLUDE_PATH', "").split(":") + [np.get_include()]  + EIGEN_INCLUDE + ["include"]

    LD_LIBRARY_PATH = [i for i in LD_LIBRARY_PATH if i]
    libraries = []

    if platform.system() == 'Windows':
        USE_OPENMP = False
        cflags_default = "-static -O3 -Wall -fPIC"
        extra_link_args_CPP = ["-std=c++11 -static", "-static-libgfortran", "-static-libgcc"]
    elif platform.system() == 'Darwin':
        cflags_default = "-O3 -Wall -fPIC -shared -Xpreprocessor -fopenmp -lomp -mmacosx-version-min=10.9"
        libraries += ["omp"]
        extra_link_args_CPP = ['-Xpreprocessor -fopenmp -lomp']
        extra_include = ['/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/']
        C_INCLUDE_PATH += ['/usr/local/opt/libomp/include']
        LD_LIBRARY_PATH += ['/usr/local/opt/libomp/lib']
    else:
        cflags_default = "-O3 -Wall -fPIC -shared -fopenmp"
        extra_link_args_CPP = ['-fopenmp']

    CFLAGS = os.environ.get('CFLAGS', cflags_default).split() + ['-I{}'.format(np.get_include())]

    USE_LAPACK = False
    if os.environ.get("USE_LAPACK", ""):
        USE_LAPACK = True
        CFLAGS += [" -llapacke -DLAPACKE=1"]
        libraries += ["lapacke"]
        extra_link_args_CPP[0] += " -llapacke"

    if os.environ.get("USE_OPENBLAS", ""):
        USE_LAPACK = True
        CFLAGS += [" -lopenblas -DLAPACKE=1"]
        libraries += ["openblas"]
        extra_link_args_CPP[0] += " -lopenblas"

    extensions = cythonize([
            Extension("libwalrus",
                sources=["thewalrus/libwalrus."+ext,],
                depends=["include/libwalrus.hpp",
                         "include/eigenvalue_hafnian.hpp",
                         "include/recursive_hafnian.hpp",
                         "include/repeated_hafnian.hpp",
                         "include/hafnian_approx.hpp",
                         "include/torontonian.hpp",
                         "include/permanent.hpp",
                         "include/hermite_multidimensional.hpp",
                         "include/stdafx.h",
                         "include/fsum.hpp"],
                include_dirs=C_INCLUDE_PATH,
                library_dirs=['/usr/lib', '/usr/local/lib'] + LD_LIBRARY_PATH,
                libraries=libraries,
                language="c++",
                extra_compile_args=["-std=c++11"] + CFLAGS,
                extra_link_args=extra_link_args_CPP)
    ], compile_time_env={'_OPENMP': USE_OPENMP, 'LAPACKE': USE_LAPACK})
else:
    extensions = []


info = {
    'name': 'thewalrus',
    'version': version,
    'maintainer': 'Xanadu Inc.',
    'maintainer_email': 'nicolas@xanadu.ai',
    'url': 'https://github.com/XanaduAI/thewalrus',
    'license': 'Apache License 2.0',
    'packages': [
                    'thewalrus',
                    'thewalrus.tests'
                ],
    'description': 'Open source library for hafnian calculation',
    'long_description': open('README.rst').read(),
    'provides': ["thewalrus"],
    'install_requires': requirements,
    'setup_requires': setup_requirements,
    'ext_modules': extensions,
    'ext_package': 'thewalrus'
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
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))
