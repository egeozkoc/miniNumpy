cmake_minimum_required(VERSION 3.16)

project(miniNumpy VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(miniNumpy INTERFACE)

target_include_directories(miniNumpy INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(basic_usage
    examples/basic_usage.cpp
)

target_link_libraries(basic_usage PRIVATE miniNumpy)

add_executable(array2d_usage
    examples/array2d_usage.cpp
)

target_link_libraries(array2d_usage PRIVATE miniNumpy)

add_executable(test_array1d
    tests/test_array1d.cpp
)

target_link_libraries(test_array1d PRIVATE miniNumpy)

add_executable(test_array2d
    tests/test_array2d.cpp
)

target_link_libraries(test_array2d PRIVATE miniNumpy)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(miniNumpy_python
    bindings/python_bindings.cpp
)

target_link_libraries(miniNumpy_python PRIVATE miniNumpy)

set_target_properties(miniNumpy_python PROPERTIES
    OUTPUT_NAME "miniNumpy"
)

enable_testing()

add_test(NAME test_array1d COMMAND test_array1d)
add_test(NAME test_array2d COMMAND test_array2d)

add_test(
    NAME test_python_bindings
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=$<TARGET_FILE_DIR:miniNumpy_python>
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_bindings.py
)