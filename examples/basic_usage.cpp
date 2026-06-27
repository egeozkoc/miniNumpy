#include <iostream>
#include "miniNumpy/Array1D.hpp"
#include "miniNumpy/functions.hpp"

int main() {
    mnp::Array1D a = {1.0, 2.0, 3.0};
    mnp::Array1D b = {4.0, 5.0, 6.0};

    mnp::Array1D c = a + b;
    mnp::Array1D d = a - b;
    mnp::Array1D e = a * b;
    mnp::Array1D f = a / b;

    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;

    std::cout << "a + b = " << c << std::endl;
    std::cout << "a - b = " << d << std::endl;
    std::cout << "a * b = " << e << std::endl;
    std::cout << "a / b = " << f << std::endl;

    std::cout << "a + 2 = " << a + 2.0 << std::endl;
    std::cout << "a - 2 = " << a - 2.0 << std::endl;
    std::cout << "a * 2 = " << a * 2.0 << std::endl;
    std::cout << "a / 2 = " << a / 2.0 << std::endl;

    std::cout << "2 + a = " << 2.0 + a << std::endl;
    std::cout << "2 - a = " << 2.0 - a << std::endl;
    std::cout << "2 * a = " << 2.0 * a << std::endl;
    std::cout << "2 / a = " << 2.0 / a << std::endl;

    std::cout << "sum(a) = " << a.sum() << std::endl;
    std::cout << "mean(a) = " << a.mean() << std::endl;
    std::cout << "min(a) = " << a.min() << std::endl;
    std::cout << "max(a) = " << a.max() << std::endl;

    std::cout << "Size: " << a.size() << std::endl;

    std::cout << "First element: " << a[0] << std::endl;

    a[1] = 10.0;

    std::cout << "Modified second element: " << a[1] << std::endl;

    std::cout << a << std::endl;

    std::cout << mnp::zeros(5) << std::endl;
    std::cout << mnp::ones(5) << std::endl;
    std::cout << "arange(0, 5) = " << mnp::arange(0, 5) << std::endl;
    std::cout << "arange(0, 1, 0.25) = " << mnp::arange(0, 1, 0.25) << std::endl;
    std::cout << "arange(5, 0, -1) = " << mnp::arange(5, 0, -1) << std::endl;


    auto x = mnp::array({1.0, 2.0, 3.0});
    auto y = mnp::ones(3);

    std::cout << "x = " << x << std::endl;
    std::cout << "y = " << y << std::endl;
    std::cout << "x + y = " << x + y << std::endl;



    return 0;
}