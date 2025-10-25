cmake_minimum_required(VERSION 3.14)
project(LinearRegression)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/b2b9072ecbe874f5937054653ef8f2731eb0f010.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(matrix-library)

add_library(LinearRegression STATIC linear_regression.cc)
target_link_libraries(LinearRegression PUBLIC MatrixLibraryAPI)
target_include_directories(LinearRegression PUBLIC ${CMAKE_SOURCE_DIR})

# Testing
include(GoogleTest)

add_executable(LinearRegressionTests linear_regression_test.cc)
target_include_directories(LinearRegressionTests PRIVATE ${CMAKE_SOURCE_DIR}/matrix-library/src)
target_link_libraries(LinearRegressionTests PRIVATE MatrixLibraryAPI gtest_main)
gtest_discover_tests(LinearRegressionTests)
