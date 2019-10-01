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
#include <iostream>
#include <complex>
#include <vector>
#include <libwalrus.hpp>


int main() {
    int nmax = 20;

    for (int m = 10; m <= nmax; m++) {
        // create a 2m*2m all ones matrix
        int n = 2 * m;
        std::vector<std::complex<double>> mat(n * n, 1.0);

        // calculate the hafnian
        std::complex<double> hafval = libwalrus::hafnian(mat);
        // print out the result
        std::cout << n << " " << hafval << std::endl;
    }

    return 0;
};
