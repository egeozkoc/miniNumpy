# miniNumpy

`miniNumpy` is a small NumPy-like array library written in C++.

The goal of this project is mainly to practice modern C++ concepts such as classes, operator overloading, header-only libraries, CMake, testing, and Python bindings (pybind11).

## Current features

* 1D arrays with `Array1D`
* 2D arrays with `Array2D`
* Elementwise arithmetic
* Scalar arithmetic
* Basic reductions: `sum`, `mean`, `min`, `max`
* Factory functions such as `zeros`, `ones`, and `arange`
* Basic reshape, flatten, transpose, and matrix multiplication
* Simple tests with `assert`
* Basic Python bindings using `pybind11`

## Example C++ usage

```cpp
#include <iostream>

#include "miniNumpy/functions.hpp"

int main() {
    auto a = mnp::arange(0, 5);
    auto b = mnp::ones(5);

    std::cout << a << std::endl;
    std::cout << a + b << std::endl;
    std::cout << a.mean() << std::endl;

    auto x = mnp::reshape(a, 1, 5);
    std::cout << x << std::endl;

    return 0;
}
```

## Example Python usage

```python
import miniNumpy as mnp

a = mnp.arange(0, 5)

print(a)
print(a.sum())
print(a.mean())

a[1] = 10
print(a)
```

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
ctest
```
