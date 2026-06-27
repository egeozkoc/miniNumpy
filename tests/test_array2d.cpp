#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>

#include "miniNumpy/Array2D.hpp"
#include "miniNumpy/functions.hpp"

bool almost_equal(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

void test_construction() {
    mnp::Array2D a(2, 3);

    assert(a.rows() == 2);
    assert(a.cols() == 3);
    assert(a.size() == 6);
    assert(!a.empty());
}

void test_indexing() {
    mnp::Array2D a(2, 3);

    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(0, 2) = 3.0;
    a(1, 0) = 4.0;
    a(1, 1) = 5.0;
    a(1, 2) = 6.0;

    assert(almost_equal(a(0, 0), 1.0));
    assert(almost_equal(a(0, 1), 2.0));
    assert(almost_equal(a(0, 2), 3.0));
    assert(almost_equal(a(1, 0), 4.0));
    assert(almost_equal(a(1, 1), 5.0));
    assert(almost_equal(a(1, 2), 6.0));
}

void test_printing() {
    mnp::Array2D a(2, 3);

    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(0, 2) = 3.0;

    a(1, 0) = 4.0;
    a(1, 1) = 5.0;
    a(1, 2) = 6.0;

    std::ostringstream oss;
    oss << a;

    assert(oss.str() == "[[1,2,3],\n [4,5,6]]");
}
void test_at() {
    mnp::Array2D a(2, 3);

    a.at(1, 2) = 10.0;
    assert(almost_equal(a.at(1, 2), 10.0));

    bool caught_exception = false;

    try {
        a.at(10, 0);
    } catch (const std::out_of_range& error) {
        caught_exception = true;
    }

    assert(caught_exception);
}

void test_initializer_list() {
    mnp::Array2D a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    assert(a.rows() == 2);
    assert(a.cols() == 3);
    assert(a.size() == 6);

    assert(almost_equal(a(0, 0), 1.0));
    assert(almost_equal(a(0, 1), 2.0));
    assert(almost_equal(a(0, 2), 3.0));
    assert(almost_equal(a(1, 0), 4.0));
    assert(almost_equal(a(1, 1), 5.0));
    assert(almost_equal(a(1, 2), 6.0));
}


void test_initializer_list_invalid_shape() {
    bool caught_exception = false;

    try {
        mnp::Array2D a = {
            {1.0, 2.0},
            {3.0}
        };
    } catch (const std::invalid_argument& error) {
        caught_exception = true;
    }

    assert(caught_exception);
}

void test_array_arithmetic() {
    mnp::Array2D a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    mnp::Array2D b = {
        {10.0, 20.0, 30.0},
        {40.0, 50.0, 60.0}
    };

    mnp::Array2D add = a + b;
    assert(almost_equal(add(0, 0), 11.0));
    assert(almost_equal(add(0, 1), 22.0));
    assert(almost_equal(add(0, 2), 33.0));
    assert(almost_equal(add(1, 0), 44.0));
    assert(almost_equal(add(1, 1), 55.0));
    assert(almost_equal(add(1, 2), 66.0));

    mnp::Array2D sub = b - a;
    assert(almost_equal(sub(0, 0), 9.0));
    assert(almost_equal(sub(0, 1), 18.0));
    assert(almost_equal(sub(0, 2), 27.0));
    assert(almost_equal(sub(1, 0), 36.0));
    assert(almost_equal(sub(1, 1), 45.0));
    assert(almost_equal(sub(1, 2), 54.0));

    mnp::Array2D mul = a * b;
    assert(almost_equal(mul(0, 0), 10.0));
    assert(almost_equal(mul(0, 1), 40.0));
    assert(almost_equal(mul(0, 2), 90.0));
    assert(almost_equal(mul(1, 0), 160.0));
    assert(almost_equal(mul(1, 1), 250.0));
    assert(almost_equal(mul(1, 2), 360.0));

    mnp::Array2D div = b / a;
    assert(almost_equal(div(0, 0), 10.0));
    assert(almost_equal(div(0, 1), 10.0));
    assert(almost_equal(div(0, 2), 10.0));
    assert(almost_equal(div(1, 0), 10.0));
    assert(almost_equal(div(1, 1), 10.0));
    assert(almost_equal(div(1, 2), 10.0));
}

void test_array_arithmetic_shape_mismatch() {
    mnp::Array2D a(2, 3);
    mnp::Array2D b(3, 2);

    bool caught_exception = false;

    try {
        mnp::Array2D c = a + b;
    } catch (const std::invalid_argument& error) {
        caught_exception = true;
    }

    assert(caught_exception);
}

void test_scalar_arithmetic() {
    mnp::Array2D a = {
        {1.0, 2.0, 4.0},
        {5.0, 10.0, 20.0}
    };

    mnp::Array2D add = a + 2.0;
    assert(almost_equal(add(0, 0), 3.0));
    assert(almost_equal(add(0, 1), 4.0));
    assert(almost_equal(add(0, 2), 6.0));
    assert(almost_equal(add(1, 0), 7.0));
    assert(almost_equal(add(1, 1), 12.0));
    assert(almost_equal(add(1, 2), 22.0));

    mnp::Array2D sub = a - 2.0;
    assert(almost_equal(sub(0, 0), -1.0));
    assert(almost_equal(sub(0, 1), 0.0));
    assert(almost_equal(sub(0, 2), 2.0));
    assert(almost_equal(sub(1, 0), 3.0));
    assert(almost_equal(sub(1, 1), 8.0));
    assert(almost_equal(sub(1, 2), 18.0));

    mnp::Array2D mul = a * 2.0;
    assert(almost_equal(mul(0, 0), 2.0));
    assert(almost_equal(mul(0, 1), 4.0));
    assert(almost_equal(mul(0, 2), 8.0));
    assert(almost_equal(mul(1, 0), 10.0));
    assert(almost_equal(mul(1, 1), 20.0));
    assert(almost_equal(mul(1, 2), 40.0));

    mnp::Array2D div = a / 2.0;
    assert(almost_equal(div(0, 0), 0.5));
    assert(almost_equal(div(0, 1), 1.0));
    assert(almost_equal(div(0, 2), 2.0));
    assert(almost_equal(div(1, 0), 2.5));
    assert(almost_equal(div(1, 1), 5.0));
    assert(almost_equal(div(1, 2), 10.0));

    mnp::Array2D left_add = 2.0 + a;
    assert(almost_equal(left_add(0, 0), 3.0));
    assert(almost_equal(left_add(1, 2), 22.0));

    mnp::Array2D left_sub = 2.0 - a;
    assert(almost_equal(left_sub(0, 0), 1.0));
    assert(almost_equal(left_sub(0, 1), 0.0));
    assert(almost_equal(left_sub(0, 2), -2.0));
    assert(almost_equal(left_sub(1, 0), -3.0));
    assert(almost_equal(left_sub(1, 1), -8.0));
    assert(almost_equal(left_sub(1, 2), -18.0));

    mnp::Array2D left_mul = 2.0 * a;
    assert(almost_equal(left_mul(0, 0), 2.0));
    assert(almost_equal(left_mul(1, 2), 40.0));

    mnp::Array2D left_div = 2.0 / a;
    assert(almost_equal(left_div(0, 0), 2.0));
    assert(almost_equal(left_div(0, 1), 1.0));
    assert(almost_equal(left_div(0, 2), 0.5));
    assert(almost_equal(left_div(1, 0), 0.4));
    assert(almost_equal(left_div(1, 1), 0.2));
    assert(almost_equal(left_div(1, 2), 0.1));
}

void test_inplace_scalar_arithmetic() {
    mnp::Array2D a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    a += 1.0;
    assert(almost_equal(a(0, 0), 2.0));
    assert(almost_equal(a(0, 1), 3.0));
    assert(almost_equal(a(1, 0), 4.0));
    assert(almost_equal(a(1, 1), 5.0));

    a -= 1.0;
    assert(almost_equal(a(0, 0), 1.0));
    assert(almost_equal(a(0, 1), 2.0));
    assert(almost_equal(a(1, 0), 3.0));
    assert(almost_equal(a(1, 1), 4.0));

    a *= 2.0;
    assert(almost_equal(a(0, 0), 2.0));
    assert(almost_equal(a(0, 1), 4.0));
    assert(almost_equal(a(1, 0), 6.0));
    assert(almost_equal(a(1, 1), 8.0));

    a /= 2.0;
    assert(almost_equal(a(0, 0), 1.0));
    assert(almost_equal(a(0, 1), 2.0));
    assert(almost_equal(a(1, 0), 3.0));
    assert(almost_equal(a(1, 1), 4.0));
}

void test_reductions() {
    mnp::Array2D a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    assert(almost_equal(a.sum(), 21.0));
    assert(almost_equal(a.mean(), 3.5));
    assert(almost_equal(a.min(), 1.0));
    assert(almost_equal(a.max(), 6.0));
}

void test_reductions_empty_array() {
    mnp::Array2D a;

    bool caught_mean = false;
    bool caught_min = false;
    bool caught_max = false;

    try {
        a.mean();
    } catch (const std::invalid_argument& error) {
        caught_mean = true;
    }

    try {
        a.min();
    } catch (const std::invalid_argument& error) {
        caught_min = true;
    }

    try {
        a.max();
    } catch (const std::invalid_argument& error) {
        caught_max = true;
    }

    assert(caught_mean);
    assert(caught_min);
    assert(caught_max);

    assert(almost_equal(a.sum(), 0.0));
}

void test_factory_functions() {
    mnp::Array2D z = mnp::zeros(2, 3);

    assert(z.rows() == 2);
    assert(z.cols() == 3);
    assert(z.size() == 6);

    assert(almost_equal(z(0, 0), 0.0));
    assert(almost_equal(z(0, 1), 0.0));
    assert(almost_equal(z(0, 2), 0.0));
    assert(almost_equal(z(1, 0), 0.0));
    assert(almost_equal(z(1, 1), 0.0));
    assert(almost_equal(z(1, 2), 0.0));

    mnp::Array2D o = mnp::ones(2, 3);

    assert(o.rows() == 2);
    assert(o.cols() == 3);
    assert(o.size() == 6);

    assert(almost_equal(o(0, 0), 1.0));
    assert(almost_equal(o(0, 1), 1.0));
    assert(almost_equal(o(0, 2), 1.0));
    assert(almost_equal(o(1, 0), 1.0));
    assert(almost_equal(o(1, 1), 1.0));
    assert(almost_equal(o(1, 2), 1.0));

    mnp::Array2D a = mnp::array({
        {1.0, 2.0},
        {3.0, 4.0}
    });

    assert(a.rows() == 2);
    assert(a.cols() == 2);

    assert(almost_equal(a(0, 0), 1.0));
    assert(almost_equal(a(0, 1), 2.0));
    assert(almost_equal(a(1, 0), 3.0));
    assert(almost_equal(a(1, 1), 4.0));
}

void test_comparison() {
    mnp::Array2D a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    mnp::Array2D b = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    mnp::Array2D c = {
        {1.0, 2.0},
        {3.0, 5.0}
    };

    mnp::Array2D d = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    assert(a == b);
    assert(!(a != b));

    assert(a != c);
    assert(!(a == c));

    assert(a != d);
    assert(!(a == d));
}

void test_utilities() {
    mnp::Array2D empty_array;

    assert(empty_array.empty());
    assert(empty_array.size() == 0);
    assert(empty_array.rows() == 0);
    assert(empty_array.cols() == 0);

    mnp::Array2D a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    assert(!a.empty());
    assert(almost_equal(a.front(), 1.0));
    assert(almost_equal(a.back(), 6.0));

    a.front() = 10.0;
    a.back() = 60.0;

    assert(almost_equal(a(0, 0), 10.0));
    assert(almost_equal(a(1, 2), 60.0));

    std::vector<double> v = a.to_vector();

    assert(v.size() == 6);
    assert(almost_equal(v[0], 10.0));
    assert(almost_equal(v[1], 2.0));
    assert(almost_equal(v[2], 3.0));
    assert(almost_equal(v[3], 4.0));
    assert(almost_equal(v[4], 5.0));
    assert(almost_equal(v[5], 60.0));
}

void test_iterators() {
    mnp::Array2D a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    double total = 0.0;

    for (double x : a) {
        total += x;
    }

    assert(almost_equal(total, 21.0));

    for (double& x : a) {
        x *= 2.0;
    }

    assert(almost_equal(a(0, 0), 2.0));
    assert(almost_equal(a(0, 1), 4.0));
    assert(almost_equal(a(0, 2), 6.0));
    assert(almost_equal(a(1, 0), 8.0));
    assert(almost_equal(a(1, 1), 10.0));
    assert(almost_equal(a(1, 2), 12.0));

    const mnp::Array2D b = {
        {1.0, 1.0},
        {2.0, 2.0}
    };

    double const_total = 0.0;

    for (double x : b) {
        const_total += x;
    }

    assert(almost_equal(const_total, 6.0));
}

void test_transpose() {
    mnp::Array2D a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    mnp::Array2D t = a.transpose();

    assert(t.rows() == 3);
    assert(t.cols() == 2);

    assert(almost_equal(t(0, 0), 1.0));
    assert(almost_equal(t(0, 1), 4.0));

    assert(almost_equal(t(1, 0), 2.0));
    assert(almost_equal(t(1, 1), 5.0));

    assert(almost_equal(t(2, 0), 3.0));
    assert(almost_equal(t(2, 1), 6.0));
}

void test_flatten() {
    mnp::Array2D a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    mnp::Array1D flat = a.flatten();

    assert(flat.size() == 6);

    assert(almost_equal(flat[0], 1.0));
    assert(almost_equal(flat[1], 2.0));
    assert(almost_equal(flat[2], 3.0));
    assert(almost_equal(flat[3], 4.0));
    assert(almost_equal(flat[4], 5.0));
    assert(almost_equal(flat[5], 6.0));
}

void test_reshape_1d_to_2d() {
    mnp::Array1D a = mnp::arange(0, 6);

    mnp::Array2D b = mnp::reshape(a, 2, 3);

    assert(b.rows() == 2);
    assert(b.cols() == 3);
    assert(b.size() == 6);

    assert(almost_equal(b(0, 0), 0.0));
    assert(almost_equal(b(0, 1), 1.0));
    assert(almost_equal(b(0, 2), 2.0));
    assert(almost_equal(b(1, 0), 3.0));
    assert(almost_equal(b(1, 1), 4.0));
    assert(almost_equal(b(1, 2), 5.0));
}

void test_reshape_size_mismatch() {
    mnp::Array1D a = mnp::arange(0, 5);

    bool caught_exception = false;

    try {
        mnp::Array2D b = mnp::reshape(a, 2, 3);
    } catch (const std::invalid_argument& error) {
        caught_exception = true;
    }

    assert(caught_exception);
}

void test_matmul() {
    mnp::Array2D a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    mnp::Array2D b = {
        {10.0, 20.0},
        {30.0, 40.0},
        {50.0, 60.0}
    };

    mnp::Array2D c = mnp::matmul(a, b);

    assert(c.rows() == 2);
    assert(c.cols() == 2);

    assert(almost_equal(c(0, 0), 220.0));
    assert(almost_equal(c(0, 1), 280.0));
    assert(almost_equal(c(1, 0), 490.0));
    assert(almost_equal(c(1, 1), 640.0));
}

void test_matmul_shape_mismatch() {
    mnp::Array2D a(2, 3);
    mnp::Array2D b(2, 3);

    bool caught_exception = false;

    try {
        mnp::Array2D c = mnp::matmul(a, b);
    } catch (const std::invalid_argument& error) {
        caught_exception = true;
    }

    assert(caught_exception);
}

int main() {
    test_construction();
    test_indexing();
    test_at();
    test_printing();
    test_initializer_list();
    test_initializer_list_invalid_shape();
    test_array_arithmetic();
    test_array_arithmetic_shape_mismatch();
    test_scalar_arithmetic();
    test_inplace_scalar_arithmetic();
    test_reductions();
    test_reductions_empty_array();
    test_factory_functions();
    test_comparison();
    test_utilities();
    test_iterators();
    test_transpose();
    test_flatten();
    test_reshape_1d_to_2d();
    test_reshape_size_mismatch();
    test_matmul();
    test_matmul_shape_mismatch();

    std::cout << "All Array2D tests passed!" << std::endl;

    return 0;
}