#include <iostream>

#include "miniNumpy/Array2D.hpp"
#include "miniNumpy/functions.hpp"


int main() {
    mnp::Array2D a(2, 3);

    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(0, 2) = 3.0;

    a(1, 0) = 4.0;
    a(1, 1) = 5.0;
    a(1, 2) = 6.0;

    std::cout << "rows = " << a.rows() << std::endl;
    std::cout << "cols = " << a.cols() << std::endl;
    std::cout << "size = " << a.size() << std::endl;

    std::cout << "a(0, 0) = " << a(0, 0) << std::endl;
    std::cout << "a(1, 2) = " << a(1, 2) << std::endl;

    std::cout << a << std::endl;

    mnp::Array2D x = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    std::cout << x << std::endl;

    mnp::Array1D i = mnp::arange(0, 6);

    mnp::Array2D j = mnp::reshape(i, 2, 3);

    std::cout << "i = " << i << std::endl;
    std::cout << "reshape(i, 2, 3) = " << j << std::endl;


    return 0;
}