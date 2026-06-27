#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "miniNumpy/Array1D.hpp"
#include "miniNumpy/functions.hpp"

bool almost_equal(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

void test_construction() {
    mnp::Array1D a = {1.0, 2.0, 3.0};

    assert(a.size() == 3);
    assert(almost_equal(a[0], 1.0));
    assert(almost_equal(a[1], 2.0));
    assert(almost_equal(a[2], 3.0));
}

void test_array_arithmetic() {
    mnp::Array1D a = {1.0, 2.0, 3.0};
    mnp::Array1D b = {4.0, 5.0, 6.0};

    mnp::Array1D c = a + b;
    assert(almost_equal(c[0], 5.0));
    assert(almost_equal(c[1], 7.0));
    assert(almost_equal(c[2], 9.0));

    mnp::Array1D d = a * b;
    assert(almost_equal(d[0], 4.0));
    assert(almost_equal(d[1], 10.0));
    assert(almost_equal(d[2], 18.0));
}

void test_scalar_arithmetic() {
    mnp::Array1D a = {1.0, 2.0, 3.0};

    mnp::Array1D b = a + 2.0;
    assert(almost_equal(b[0], 3.0));
    assert(almost_equal(b[1], 4.0));
    assert(almost_equal(b[2], 5.0));

    mnp::Array1D c = 2.0 * a;
    assert(almost_equal(c[0], 2.0));
    assert(almost_equal(c[1], 4.0));
    assert(almost_equal(c[2], 6.0));
}

void test_reductions() {
    mnp::Array1D a = {1.0, 2.0, 3.0};

    assert(almost_equal(a.sum(), 6.0));
    assert(almost_equal(a.mean(), 2.0));
    assert(almost_equal(a.min(), 1.0));
    assert(almost_equal(a.max(), 3.0));
}

void test_factory_functions() {
    mnp::Array1D z = mnp::zeros(3);
    assert(almost_equal(z[0], 0.0));
    assert(almost_equal(z[1], 0.0));
    assert(almost_equal(z[2], 0.0));

    mnp::Array1D o = mnp::ones(3);
    assert(almost_equal(o[0], 1.0));
    assert(almost_equal(o[1], 1.0));
    assert(almost_equal(o[2], 1.0));

    mnp::Array1D r = mnp::arange(0, 5);
    assert(r.size() == 5);
    assert(almost_equal(r[0], 0.0));
    assert(almost_equal(r[1], 1.0));
    assert(almost_equal(r[4], 4.0));
}

void test_at() {
    mnp::Array1D a = {1.0, 2.0, 3.0};

    assert(almost_equal(a.at(0), 1.0));
    assert(almost_equal(a.at(1), 2.0));
    assert(almost_equal(a.at(2), 3.0));

    a.at(1) = 10.0;
    assert(almost_equal(a.at(1), 10.0));

    bool caught_exception = false;

    try {
        a.at(100);
    } catch (const std::out_of_range& error) {
        caught_exception = true;
    }

    assert(caught_exception);

    const mnp::Array1D b = {4.0, 5.0, 6.0};

    assert(almost_equal(b.at(0), 4.0));
    assert(almost_equal(b.at(1), 5.0));
    assert(almost_equal(b.at(2), 6.0));

}

void test_inplace_array_arithmetic() {
    mnp::Array1D a = {1.0, 2.0, 3.0};
    mnp::Array1D b = {4.0, 5.0, 6.0};

    a += b;
    assert(almost_equal(a[0], 5.0));
    assert(almost_equal(a[1], 7.0));
    assert(almost_equal(a[2], 9.0));

    a -= b;
    assert(almost_equal(a[0], 1.0));
    assert(almost_equal(a[1], 2.0));
    assert(almost_equal(a[2], 3.0));

    a *= b;
    assert(almost_equal(a[0], 4.0));
    assert(almost_equal(a[1], 10.0));
    assert(almost_equal(a[2], 18.0));

    a /= b;
    assert(almost_equal(a[0], 1.0));
    assert(almost_equal(a[1], 2.0));
    assert(almost_equal(a[2], 3.0));
}

void test_inplace_scalar_arithmetic() {
    mnp::Array1D a = {1.0, 2.0, 3.0};

    a += 2.0;
    assert(almost_equal(a[0], 3.0));
    assert(almost_equal(a[1], 4.0));
    assert(almost_equal(a[2], 5.0));

    a -= 2.0;
    assert(almost_equal(a[0], 1.0));
    assert(almost_equal(a[1], 2.0));
    assert(almost_equal(a[2], 3.0));

    a *= 2.0;
    assert(almost_equal(a[0], 2.0));
    assert(almost_equal(a[1], 4.0));
    assert(almost_equal(a[2], 6.0));

    a /= 2.0;
    assert(almost_equal(a[0], 1.0));
    assert(almost_equal(a[1], 2.0));
    assert(almost_equal(a[2], 3.0));
}

void test_comparison() {
    mnp::Array1D a = {1.0, 2.0, 3.0};
    mnp::Array1D b = {1.0, 2.0, 3.0};
    mnp::Array1D c = {1.0, 2.0, 4.0};
    mnp::Array1D d = {1.0, 2.0};

    assert(a == b);
    assert(!(a != b));

    assert(a != c);
    assert(!(a == c));

    assert(a != d);
    assert(!(a == d));
}

void test_utilities() {
    mnp::Array1D empty_array;

    assert(empty_array.empty());
    assert(empty_array.size() == 0);

    mnp::Array1D a = {1.0, 2.0, 3.0};

    assert(!a.empty());
    assert(almost_equal(a.front(), 1.0));
    assert(almost_equal(a.back(), 3.0));

    a.front() = 10.0;
    a.back() = 30.0;

    assert(almost_equal(a[0], 10.0));
    assert(almost_equal(a[2], 30.0));

    std::vector<double> v = a.to_vector();

    assert(v.size() == 3);
    assert(almost_equal(v[0], 10.0));
    assert(almost_equal(v[1], 2.0));
    assert(almost_equal(v[2], 30.0));
}

void test_iterators() {
    mnp::Array1D a = {1.0, 2.0, 3.0};

    double total = 0.0;

    for (double x : a) {
        total += x;
    }

    assert(almost_equal(total, 6.0));

    for (double& x : a) {
        x *= 2.0;
    }

    assert(almost_equal(a[0], 2.0));
    assert(almost_equal(a[1], 4.0));
    assert(almost_equal(a[2], 6.0));

    const mnp::Array1D b = {4.0, 5.0, 6.0};

    double const_total = 0.0;

    for (double x : b) {
        const_total += x;
    }

    assert(almost_equal(const_total, 15.0));
}

void test_array_factory() {
    mnp::Array1D a = mnp::array({1.0, 2.0, 3.0});

    assert(a.size() == 3);
    assert(almost_equal(a[0], 1.0));
    assert(almost_equal(a[1], 2.0));
    assert(almost_equal(a[2], 3.0));
}

int main() {
    test_construction();
    test_array_arithmetic();
    test_scalar_arithmetic();
    test_reductions();
    test_factory_functions();
    test_at();
    test_inplace_array_arithmetic();
    test_inplace_scalar_arithmetic();
    test_comparison();
    test_utilities();
    test_iterators();
    test_array_factory();


    std::cout << "All tests passed!" << std::endl;

    return 0;
}