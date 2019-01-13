// Copyright 2018 Xanadu Quantum Technologies Inc.

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
#include <hafnian.hpp>

#include <sys/time.h>
typedef unsigned long long timestamp_t;

static timestamp_t
get_timestamp()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}


int main() {
    int nmax = 12;
    double time;
    timestamp_t t0, t1;

    t0 = get_timestamp();
    for (int m = 1; m <= nmax; m++) {
        // create a 2m*2m all ones matrix
        int n = 2*m;
        std::vector<double> mat(n*n, 1.0);

        // calculate the hafnian
        double hafval = hafnian::hafnian(mat);
        // print out the result
        std::cout << hafval << std::endl;
    }
    t1 = get_timestamp();
    time = (t1 - t0) / 1000000.0L;
    std::cout << "Time taken (hafnian):" << time << std::endl << std::endl;


    t0 = get_timestamp();
    for (int m = 1; m <= nmax; m++) {
        // create a 2m*2m all ones matrix
        int n = 2*m;
        std::vector<double> mat(n*n, 1.0);

        // calculate the hafnian
        double hafval = hafnian::hafnian_recursive(mat);
        // print out the result
        std::cout << hafval << std::endl;
    }
    t1 = get_timestamp();
    time = (t1 - t0) / 1000000.0L;
    std::cout << "Time taken (recursive):" << time << std::endl << std::endl;


    t0 = get_timestamp();
    for (int m = 1; m <= nmax; m++) {
        // create a 2m*2m all ones matrix
        int n = 2*m;
        std::vector<double> mat(1, 1.0);
        std::vector<int> rpt(1, n);

        // calculate the hafnian
        double hafval = hafnian::hafnian_rpt(mat, rpt);
        // print out the result
        std::cout << hafval << std::endl;
    }
    t1 = get_timestamp();
    time = (t1 - t0) / 1000000.0L;
    std::cout << "Time taken (rpt eigen):" << time << std::endl << std::endl;


    t0 = get_timestamp();
    for (int m = 1; m <= nmax; m++) {
        // create a 2m*2m all ones matrix
        int n = 2*m;
        std::vector<double> mat(1, 1.0);
        std::vector<int> rpt(1, n);

        // calculate the hafnian
        double hafval = hafnian::hafnian_rpt(mat, rpt, false);
        // print out the result
        std::cout << hafval << std::endl;
    }
    t1 = get_timestamp();
    time = (t1 - t0) / 1000000.0L;
    std::cout << "Time taken (rpt):" << time << std::endl << std::endl;

    return 0;
};
