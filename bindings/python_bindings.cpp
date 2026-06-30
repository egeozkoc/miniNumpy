#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "miniNumpy/Array1D.hpp"
#include "miniNumpy/Array2D.hpp"
#include "miniNumpy/functions.hpp"

namespace py = pybind11;

PYBIND11_MODULE(miniNumpy, m) {
    m.doc() = "A small NumPy-like library written in C++";

    py::class_<mnp::Array1D>(m, "Array1D")
        .def(py::init<>())
        .def(py::init<std::size_t>())
        .def("size", &mnp::Array1D::size)
        .def("empty", &mnp::Array1D::empty)
        .def("sum", &mnp::Array1D::sum)
        .def("mean", &mnp::Array1D::mean)
        .def("min", &mnp::Array1D::min)
        .def("max", &mnp::Array1D::max)
        .def("__repr__", [](const mnp::Array1D& arr) {
            std::ostringstream oss;
            oss << arr;
            return oss.str();
        })
        .def("__add__", [](const mnp::Array1D& a, const mnp::Array1D& b) {
            return a + b;
        })
        .def("__sub__", [](const mnp::Array1D& a, const mnp::Array1D& b) {
            return a - b;
        })
        .def("__mul__", [](const mnp::Array1D& a, const mnp::Array1D& b) {
            return a * b;
        })
        .def("__truediv__", [](const mnp::Array1D& a, const mnp::Array1D& b) {
            return a / b;
        })
        .def("__add__", [](const mnp::Array1D& a, double scalar) {
            return a + scalar;
        })
        .def("__sub__", [](const mnp::Array1D& a, double scalar) {
            return a - scalar;
        })
        .def("__mul__", [](const mnp::Array1D& a, double scalar) {
            return a * scalar;
        })
        .def("__truediv__", [](const mnp::Array1D& a, double scalar) {
            return a / scalar;
        })
        .def("__radd__", [](const mnp::Array1D& a, double scalar) {
            return scalar + a;
        })
        .def("__rsub__", [](const mnp::Array1D& a, double scalar) {
            return scalar - a;
        })
        .def("__rmul__", [](const mnp::Array1D& a, double scalar) {
            return scalar * a;
        })
        .def("__rtruediv__", [](const mnp::Array1D& a, double scalar) {
            return scalar / a;
        })
        .def("__getitem__", [](const mnp::Array1D& arr, py::ssize_t index) {
            if (index < 0) {
                index += static_cast<py::ssize_t>(arr.size());
            }

            if (index < 0 || static_cast<std::size_t>(index) >= arr.size()) {
                throw py::index_error("Array1D index out of range");
            }

            return arr[static_cast<std::size_t>(index)];
        })
        .def("__setitem__", [](mnp::Array1D& arr, py::ssize_t index, double value) {
            if (index < 0) {
                index += static_cast<py::ssize_t>(arr.size());
            }

            if (index < 0 || static_cast<std::size_t>(index) >= arr.size()) {
                throw py::index_error("Array1D index out of range");
            }

            arr[static_cast<std::size_t>(index)] = value;
        })
        .def("__len__", &mnp::Array1D::size)
        .def_property_readonly("shape", [](const mnp::Array1D& arr) {
            return py::make_tuple(arr.size());
        });

    py::class_<mnp::Array2D>(m, "Array2D")
        .def(py::init<>())
        .def(py::init<std::size_t, std::size_t>())
        .def("rows", &mnp::Array2D::rows)
        .def("cols", &mnp::Array2D::cols)
        .def("size", &mnp::Array2D::size)
        .def("empty", &mnp::Array2D::empty)
        .def("sum", &mnp::Array2D::sum)
        .def("mean", &mnp::Array2D::mean)
        .def("min", &mnp::Array2D::min)
        .def("max", &mnp::Array2D::max)
        .def("transpose", &mnp::Array2D::transpose)
        .def("row", &mnp::Array2D::row)
        .def("col", &mnp::Array2D::col)
        .def("flatten", &mnp::Array2D::flatten)
        .def("__repr__", [](const mnp::Array2D& arr) {
            std::ostringstream oss;
            oss << arr;
            return oss.str();
        })
        .def("__getitem__", [](const mnp::Array2D& arr, std::pair<py::ssize_t, py::ssize_t> index) {
            py::ssize_t row = index.first;
            py::ssize_t col = index.second;

            if (row < 0) {
                row += static_cast<py::ssize_t>(arr.rows());
            }

            if (col < 0) {
                col += static_cast<py::ssize_t>(arr.cols());
            }

            if (row < 0 || static_cast<std::size_t>(row) >= arr.rows() ||
                col < 0 || static_cast<std::size_t>(col) >= arr.cols()) {
                throw py::index_error("Array2D index out of range");
            }

            return arr(
                static_cast<std::size_t>(row),
                static_cast<std::size_t>(col)
            );
        })
        .def("__setitem__", [](mnp::Array2D& arr, std::pair<py::ssize_t, py::ssize_t> index, double value) {
            py::ssize_t row = index.first;
            py::ssize_t col = index.second;

            if (row < 0) {
                row += static_cast<py::ssize_t>(arr.rows());
            }

            if (col < 0) {
                col += static_cast<py::ssize_t>(arr.cols());
            }

            if (row < 0 || static_cast<std::size_t>(row) >= arr.rows() ||
                col < 0 || static_cast<std::size_t>(col) >= arr.cols()) {
                throw py::index_error("Array2D index out of range");
            }

            arr(
                static_cast<std::size_t>(row),
                static_cast<std::size_t>(col)
            ) = value;
        })
        .def_property_readonly("shape", [](const mnp::Array2D& arr) {
            return py::make_tuple(arr.rows(), arr.cols());
        });

    m.def("zeros", py::overload_cast<std::size_t>(&mnp::zeros));
    m.def("ones", py::overload_cast<std::size_t>(&mnp::ones));

    m.def(
        "arange",
        &mnp::arange,
        py::arg("start"),
        py::arg("stop"),
        py::arg("step") = 1.0
    );

    m.def("zeros2d", py::overload_cast<std::size_t, std::size_t>(&mnp::zeros));
    m.def("ones2d", py::overload_cast<std::size_t, std::size_t>(&mnp::ones));

    m.def("reshape", &mnp::reshape);
    m.def("matmul", &mnp::matmul);
}