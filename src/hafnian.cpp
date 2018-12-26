#include <hafnian.hpp>


int main() {
    int n = 32;
    std::cout << n << std::endl;
    std::vector<long long int> z(n*n, 0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i < n/2) {
                if (j < n/2) {
                    z[i*n+j] = 0;
                }
                else {
                    z[i*n+j] = 1;
                }
            }
            else {
                if (j < n/2) {
                    z[i*n+j] = 1;
                }
                else {
                    z[i*n+j] = 0;
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << z[i*n+j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    long long int haf = hafnian_int(z);

    std::cout << haf << std::endl;

    return 0;
};
