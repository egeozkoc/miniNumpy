#pragma once

#include <stdexcept>
#include <cstddef>
#include <initializer_list>

#include "miniNumpy/Array1D.hpp"
#include "miniNumpy/Array2D.hpp"

namespace mnp {

    inline Array1D array(std::initializer_list<double> values) {
        return Array1D(values);
    }

    inline Array1D zeros(std::size_t size) {
        return Array1D(size);

    }

    inline Array1D ones(std::size_t size) {
        Array1D result(size);

        for (std::size_t i = 0; i < size; ++i) {
            result[i] = 1.0;
        }

        return result;
    }

    inline Array1D arange(double start, double stop, double step = 1.0) {
        if (step == 0.0) {
            throw std::invalid_argument("Step cannot be zero");
        }

        std::size_t count = 0;

        if (step > 0) {
            for (double value = start; value < stop; value += step) {
                ++count;
            }
        } else {
            for (double value = start; value > stop; value += step) {
                ++count;
            }
        }

        Array1D result(count);

        std::size_t i = 0;

        if (step > 0) {
            for (double value = start; value < stop; value += step) {
                result[i] = value;
                ++i;
            }
        } else {
            for (double value = start; value > stop; value += step) {
                result[i] = value;
                ++i;
            }
        }

        return result;
    }


    inline Array2D array(std::initializer_list<std::initializer_list<double>> values) {
        return Array2D(values);
    }

    inline Array2D zeros(std::size_t rows, std::size_t cols) {
        return Array2D(rows, cols);
}

    inline Array2D ones(std::size_t rows, std::size_t cols) {
        Array2D result(rows, cols);

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                result(i, j) = 1.0;
            }
        }

        return result;
    }

    inline Array2D reshape(const Array1D& arr, std::size_t rows, std::size_t cols) {
        if (arr.size() != rows * cols) {
            throw std::invalid_argument("Cannot reshape array: size mismatch");
        }

        Array2D result(rows, cols);

        for (std::size_t i = 0; i < arr.size(); ++i) {
            std::size_t row = i / cols;
            std::size_t col = i % cols;

            result(row, col) = arr[i];
        }

        return result;
    }

    inline Array2D matmul(const Array2D& a, const Array2D& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument(
            "Cannot multiply matrices: left columns must match right rows"
        );
    }

    Array2D result(a.rows(), b.cols());

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            double sum = 0.0;

            for (std::size_t k = 0; k < a.cols(); ++k) {
                sum += a(i, k) * b(k, j);
            }

            result(i, j) = sum;
        }
    }

    return result;
}


} // namespace mnp