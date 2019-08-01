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
"""This module performs benchmarking on the Python interface lhaf"""
import time

import numpy as np
from thewalrus import haf_complex

header = ["Size", "Time(complex128)", "Result(complex128)"]

print("{: >5} {: >15} {: >25} ".format(*header))


for n in range(2, 23):
    mat2 = np.ones([2*n, 2*n], dtype=np.complex128)
    init2 = time.clock()
    x2 = np.real(haf_complex(mat2))
    end2 = time.clock()
    row = [2*n, end2-init2, x2]
    print("{: >5} {: >15} {: >25}".format(*row))
