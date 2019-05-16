// MIT License

// Copyright (c) 2018 Albert Chan

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once
#include <math.h>

namespace fsum {
  class sc_partials {         // shewchuk algorithm
    public:
      sc_partials() {*this = 0.0;}
      void operator+=(double x);
      void operator-=(double x) {return operator+=(-x);}
      void operator=(double x)  {sum[last = 0] = x;}
      operator double() const;

      int last;
      double sum[SC_STACK];
  };

  void sc_partials::operator+=(double x) {
  COMPRESS_STACK:;
    int i=0;
    double y, hi, lo;
    for(int j=0; j <= last; j++) {
      y = sum[j];
      hi = x + y;
      lo = (std::fabs(x) < std::fabs(y)) ? x - (hi - y) : y - (hi - x);
      x = hi;
      if (lo) sum[i++] = lo;      // save partials
    }
    if (!i || !std::isfinite(x)) {sum[ last = 0 ] = x; return;}
    sum[ last = i ] = x;
    if (i == SC_STACK - 1) {x = 0.0; goto COMPRESS_STACK;}
  }

  sc_partials::operator double() const {
    int i = last;
    if (i == 0) return sum[0];
    double lo, hi, x = sum[i];
    do {
      lo = sum[--i];              // sum in reverse
      hi = x + lo;
      lo -= (hi - x);
      x = hi;
    }
    while (i && lo == 0);
    if (i && (lo < 0) == (sum[i-1] < 0))
      if ((hi = x + (lo *= 2)), (lo == (hi - x)))
        x = hi;                   // half-way case
    return x;
  }
}
