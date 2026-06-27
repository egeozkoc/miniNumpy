#pragma once

#include <vector>
#include <initializer_list>
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace mnp {

    class Array1D{
        private:
        std::vector<double> data_;

        public:
        Array1D() = default;

        Array1D(std::size_t size)
            : data_(size) {}

        Array1D(std::initializer_list<double> values)
            : data_(values) {}

        std::size_t size() const {
            return data_.size();            
        }

        double& operator[](std::size_t index) {
            return data_[index];

        }

        const double& operator[](std::size_t index) const {
            return data_[index];
        }

        double& at(std::size_t index) {
            return data_.at(index);
        }

        const double& at(std::size_t index) const {
            return data_.at(index);
        }

        Array1D& operator+=(const Array1D& other) {
            if (size() != other.size()) {
                throw std::invalid_argument("Array sizes must match for addition");
            }

            for (std::size_t i = 0; i < size(); ++i) {
                data_[i] += other[i];

            }

            return *this;
        }

        Array1D& operator-=(const Array1D& other) {
            if (size() != other.size()) {
                throw std::invalid_argument("Array sizes must match for addition");
            }

            for (std::size_t i = 0; i < size(); ++i) {
                data_[i] -= other[i];

            }

            return *this;
        }

        Array1D& operator*=(const Array1D& other) {
            if (size() != other.size()) {
                throw std::invalid_argument("Array sizes must match for addition");
            }

            for (std::size_t i = 0; i < size(); ++i) {
                data_[i] *= other[i];

            }

            return *this;
        }

        Array1D& operator/=(const Array1D& other) {
            if (size() != other.size()) {
                throw std::invalid_argument("Array sizes must match for addition");
            }

            for (std::size_t i = 0; i < size(); ++i) {
                data_[i] /= other[i];

            }

            return *this;
        }

        Array1D& operator+=(double scalar) {
            for (std::size_t i = 0; i < size(); ++i) {
                data_[i] += scalar;

            }

            return *this;
        }

        Array1D& operator-=(double scalar) {
            for (std::size_t i = 0; i < size(); ++i) {
                data_[i] -= scalar;

            }

            return *this;
        }

        Array1D& operator*=(double scalar) {
            for (std::size_t i = 0; i < size(); ++i) {
                data_[i] *= scalar;

            }

            return *this;
        }

        Array1D& operator/=(double scalar) {
            for (std::size_t i = 0; i < size(); ++i) {
                data_[i] /= scalar;

            }

            return *this;
        }

        Array1D operator+(const Array1D& other) const {
            Array1D result = *this;
            result += other;
            return result;

        }
        
        Array1D operator-(const Array1D& other) const {
            Array1D result = *this;
            result -= other;
            return result;

        }
        Array1D operator*(const Array1D& other) const {
            Array1D result = *this;
            result *= other;
            return result;

        }

        Array1D operator/(const Array1D& other) const {
            Array1D result = *this;
            result /= other;
            return result;

        }

        Array1D operator+(double scalar) const {
            Array1D result = *this;
            result += scalar;
            return result;
        }

        Array1D operator-(double scalar) const {
            Array1D result = *this;
            result -= scalar;
            return result;
        }

        Array1D operator*(double scalar) const {
            Array1D result = *this;
            result *= scalar;
            return result;
        }

        Array1D operator/(double scalar) const {
            Array1D result = *this;
            result /= scalar;
            return result;
        }

        double sum() const {
            return std::accumulate(data_.begin(), data_.end(), 0.0);
        }

        double mean() const {
            if (size() == 0) {
                throw std::invalid_argument("Cannot compute mean of an empty array");
            }

            return sum() / static_cast<double>(size());
        }

        double min() const {
            if(size() == 0) {
                throw std::invalid_argument("Cannot compute min of an empty array");
            }

            return *std::min_element(data_.begin(), data_.end());
        }

        double max() const {
            if (size() == 0) {
                throw std::invalid_argument("Cannot compute max of an empty array");
            }

            return *std::max_element(data_.begin(), data_.end());
        }

        bool operator==(const Array1D& other) const {
            if (size() != other.size()) {
                return false;
            }

            for (std::size_t i=0; i < size(); ++i){
                if (data_[i] != other[i]) {
                    return false;
                }
            }
            
            return true;
        }

        bool operator!=(const Array1D& other) const {
            return !(*this == other);

        }

        bool empty() const {
            return data_.empty();
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
    };

    inline Array1D operator+(double scalar, const Array1D& arr) {
        return arr + scalar;
    }

    inline Array1D operator*(double scalar, const Array1D& arr) {
        return arr * scalar;
    }

    inline Array1D operator-(double scalar, const Array1D& arr) {
        Array1D result(arr.size());

        for (std::size_t i = 0; i < arr.size(); ++i) {
            result[i] = scalar - arr[i];
        }
        
        return result;
    }

    inline Array1D operator/(double scalar, const Array1D& arr) {
        Array1D result(arr.size());

        for (std::size_t i = 0; i < arr.size(); ++i) {
            result[i] = scalar / arr[i];
        }
        
        return result;
    }

    

    inline std::ostream& operator<<(std::ostream& os, const Array1D& arr) {
        os << "[";

        for (std::size_t i=0; i<arr.size(); ++i){

            os << arr[i];

            if (i + 1 < arr.size()) {
                os << ",";
            }
        }

        os << "]";
        return os;
    }

}

