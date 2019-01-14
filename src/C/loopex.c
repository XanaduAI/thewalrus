// Copyright 2019 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <stdio.h>
#include "lhafnian.h"


int main()
{
    double complex mat[16] =
    {
        0.52712360 + 1.13707076 * I, 0.75772661 + 1.02491658 * I,
        0.38518232 + 1.1731741 * I,  1.99073280 + 1.01656828 * I,
        0.75772661 + 1.02491658 * I, 1.55018796 + 0.57571399 * I,
        0.80696518 + 0.8125917 * I,  1.13377451 + 0.94711957 * I,
        0.38518232 + 1.1731741 * I,   0.80696518 + 0.8125917 * I,
        1.81015953 + 1.06267482 * I, 1.18484893 + 0.65869646 * I,
        1.99073280 + 1.01656828 * I, 1.13377451 + 0.94711957 * I,
        1.18484893 + 0.65869646 * I, 1.46259098 + 1.15082374 * I
    };

    double complex hafval = hafnian_loops(mat, 4);

    printf(" %lf %lf\n", creal(hafval), cimag(hafval));

    return 0;
}
