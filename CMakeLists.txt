# cd build
# cmake .. 
# cmake --build .
# 
# cmake --build . --target clean
cmake_minimum_required(VERSION 3.18)

project(
    3dPathTracer
    VERSION 3.0.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED) #
find_package(OPTIX90 REQUIRED)     # gets OptiX_INCLUDE
find_package(Pathtracer REQUIRED)  # gets PATHTRACER_INCLUDE_DIR PATHTRACER_LIB

add_executable(3dPathTracerApp "appMain.cpp")

set_property(TARGET 3dPathTracerApp PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(
    3dPathTracerApp
    PRIVATE
        PATHTRACER::pathtracer # we dont need any device side functions because were just calling lib functions
)

target_include_directories(
    3dPathTracerApp 
    PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
)

include(CTest)
enable_testing()

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/*.cpp" "${CMAKE_SOURCE_DIR}/tests/*.cu")


add_executable(3dPathTracerTests ${TEST_SOURCES})
target_link_libraries(
    3dPathTracerTests 
    PUBLIC
        gtest_main
        CUDA::cudart
        PATHTRACER::pathtracer
)

target_include_directories(
    3dPathTracerTests
    PUBLIC
        "${CMAKE_SOURCE_DIR}/tests"
)