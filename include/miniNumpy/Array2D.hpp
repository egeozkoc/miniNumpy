# pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <ostream>
#include <initializer_list>
#include <string>
#include <numeric>
#include <algorithm>

#include "miniNumpy/Array1D.hpp"

namespace mnp {

    class Array2D {
        private:
            std::size_t rows_;
            std::size_t cols_;
            std::vector<double> data_;

            void check_same_shape(const Array2D& other, const char* operation) const {
                if (rows_ != other.rows_ || cols_ != other.cols_) {
                    throw std::invalid_argument(
                        std::string("Array shapes must match for ") + operation
                    );
                }
            }

        public:
            Array2D()
                : rows_(0), cols_(0), data_() {}

            Array2D(std::size_t rows, std::size_t cols)
                : rows_(rows), cols_(cols), data_(rows * cols) {}

            Array2D(std::initializer_list<std::initializer_list<double>> values)
                : rows_(values.size()),
                  cols_(values.size() == 0 ? 0 : values.begin()->size()),
                  data_()
                {
                    data_.reserve(rows_ * cols_);

                    for (const auto& row : values) {
                        if (row.size() != cols_) {
                            throw std::invalid_argument("All rows must have the same number of columns");
                        }

                        for (double value : row) {
                            data_.push_back(value);
                        }
                    }
                }

            std::size_t rows() const {
                return rows_;
            }

            std::size_t cols() const {
                return cols_;
            }

            std::size_t size() const {
                return data_.size();
            }

            bool empty() const {
                return data_.empty();
            }

            double& operator()(std::size_t row, std::size_t col) {
                return data_[row * cols_ + col];
            }

            const double& operator()(std::size_t row, std::size_t col) const {
                return data_[row * cols_ + col];
            }

            double& at(std::size_t row, std::size_t col) {
                if (row >= rows_ || col >= cols_) {
                    throw std::out_of_range("Array2D index out of range");
                }

                return data_[row * cols_ + col];
            }

            const double& at(std::size_t row, std::size_t col) const {
            if (row >= rows_ || col >= cols_) {
                    throw std::out_of_range("Array2D index out of range");
                }

                return data_[row * cols_ + col];
            }

            Array2D& operator+=(const Array2D& other) {
                check_same_shape(other, "addition");

                for (std::size_t i = 0; i < size(); ++i) {
                    data_[i] += other.data_[i];
                }

                return *this;
            }

            Array2D& operator-=(const Array2D& other) {
                check_same_shape(other, "subtraction");

                for (std::size_t i = 0; i < size(); ++i) {
                    data_[i] -= other.data_[i];
                }

                return *this;
            }

            Array2D& operator*=(const Array2D& other) {
                check_same_shape(other, "multiplication");

                for (std::size_t i = 0; i < size(); ++i) {
                    data_[i] *= other.data_[i];
                }

                return *this;
            }

            Array2D& operator/=(const Array2D& other) {
                check_same_shape(other, "division");

                for (std::size_t i = 0; i < size(); ++i) {
                    data_[i] /= other.data_[i];
                }

                return *this;
            }

            Array2D operator+(const Array2D& other) const {
                Array2D result = *this;
                result += other;
                return result;
            }

            Array2D operator-(const Array2D& other) const {
                Array2D result = *this;
                result -= other;
                return result;
            }

            Array2D operator*(const Array2D& other) const {
                Array2D result = *this;
                result *= other;
                return result;
            }

            Array2D operator/(const Array2D& other) const {
                Array2D result = *this;
                result /= other;
                return result;
            }

            Array2D& operator+=(double scalar) {
                for (std::size_t i = 0; i < size(); ++i) {
                    data_[i] += scalar;
                }

                return *this;
            }

            Array2D& operator-=(double scalar) {
                for (std::size_t i = 0; i < size(); ++i) {
                    data_[i] -= scalar;
                }

                return *this;
            }

            Array2D& operator*=(double scalar) {
                for (std::size_t i = 0; i < size(); ++i) {
                    data_[i] *= scalar;
                }

                return *this;
            }

            Array2D& operator/=(double scalar) {
                for (std::size_t i = 0; i < size(); ++i) {
                    data_[i] /= scalar;
                }

                return *this;
            }

            Array2D operator+(double scalar) const {
                Array2D result = *this;
                result += scalar;
                return result;
            }

            Array2D operator-(double scalar) const {
                Array2D result = *this;
                result -= scalar;
                return result;
            }

            Array2D operator*(double scalar) const {
                Array2D result = *this;
                result *= scalar;
                return result;
            }

            Array2D operator/(double scalar) const {
                Array2D result = *this;
                result /= scalar;
                return result;
            }

            double sum() const {
                return std::accumulate(data_.begin(), data_.end(), 0.0);
            }

            double mean() const {
                if (empty()) {
                    throw std::invalid_argument("Cannot compute mean of an empty Array2D");
                }

                return sum() / static_cast<double>(size());
            }

            double min() const {
                if (empty()) {
                    throw std::invalid_argument("Cannot compute min of an empty Array2D");
                }

                return *std::min_element(data_.begin(), data_.end());
            }

            double max() const {
                if (empty()) {
                    throw std::invalid_argument("Cannot compute max of an empty Array2D");
                }

                return *std::max_element(data_.begin(), data_.end());
            }

            bool operator==(const Array2D& other) const {
                if (rows_ != other.rows_ || cols_ != other.cols_) {
                    return false;
                }

                for (std::size_t i = 0; i < size(); ++i) {
                    if (data_[i] != other.data_[i]) {
                        return false;
                    }
                }

                return true;
            }

            bool operator!=(const Array2D& other) const {
                return !(*this == other);
            }

            double& front() {
                return data_.front();
            }

            const double& front() const {
                return data_.front();
            }

            double& back() {
                return data_.back();
            }

            const double& back() const {
                return data_.back();
            }

            std::vector<double> to_vector() const {
                return data_;
            }

            auto begin() {
                return data_.begin();
            }

            auto end() {
                return data_.end();
            }

            auto begin() const {
                return data_.begin();
            }

            auto end() const {
                return data_.end();
            }

            Array2D transpose() const {
                Array2D result(cols_, rows_);

                for (std::size_t i = 0; i < rows_; ++i) {
                    for (std::size_t j = 0; j < cols_; ++j) {
                        result(j, i) = (*this)(i, j);
                    }
                }

                return result;
            }

            Array1D flatten() const {
                Array1D result(size());

                for (std::size_t i = 0; i < size(); ++i) {
                    result[i] = data_[i];
                }

                return result;
            }

            
    };

    inline Array2D operator+(double scalar, const Array2D& arr) {
        return arr + scalar;
    }

    inline Array2D operator*(double scalar, const Array2D& arr) {
        return arr * scalar;
    }

    inline Array2D operator-(double scalar, const Array2D& arr) {
        Array2D result(arr.rows(), arr.cols());

        for (std::size_t i = 0; i < arr.rows(); ++i) {
            for (std::size_t j = 0; j < arr.cols(); ++j) {
                result(i, j) = scalar - arr(i, j);
            }
        }

        return result;
    }

    inline Array2D operator/(double scalar, const Array2D& arr) {
        Array2D result(arr.rows(), arr.cols());

        for (std::size_t i = 0; i < arr.rows(); ++i) {
            for (std::size_t j = 0; j < arr.cols(); ++j) {
                result(i, j) = scalar / arr(i, j);
            }
        }

        return result;
    }

    inline std::ostream& operator<<(std::ostream& os, const Array2D& arr) {
        os << "[";

        for (std::size_t i = 0; i < arr.rows(); ++i) {
            if (i > 0) {
                os << " ";
            }

            os << "[";

            for (std::size_t j = 0; j < arr.cols(); ++j) {
                os << arr(i, j);

                if (j + 1 < arr.cols()) {
                    os << ",";
                }
            }

            os << "]";

            if (i + 1 < arr.rows()) {
                os << ",\n";
            }
        }

        os << "]";

        return os;
    }
} // namespace mnp