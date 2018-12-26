#include <hafnian.hpp>


int main() {
    int n = 2;
    std::cout << n << std::endl;
    vec_complex z(n*n, 0.0);

    z[0] = std::complex<double>(2.4324, 0.12343);
    z[1] = std::complex<double>(-0.5435435, 0.21312321);
    z[2] = z[1];
    z[3] = std::complex<double>(-1.54321, -0.927345);

    double_complex expected = z[1];
    double_complex haf = hafnian(z);

    std::cout << haf << std::endl;
    std::cout << expected << std::endl;

    expected = z[1] + z[0]*z[3];
    haf = loop_hafnian(z);

    std::cout << haf << std::endl;
    std::cout << expected << std::endl;

    return 0;
};
