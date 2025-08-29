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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED) #
find_package(OPTIX90 REQUIRED)     # gets OPTIX_INCLUDE_DIR 
find_package(Pathtracer REQUIRED)  # gets PATHTRACER_INCLUDE_DIR PATHTRACER_LIB

message("CMAKE_SOURCE_DIR: " "${CMAKE_SOURCE_DIR}")

add_executable(
    3dPathTracerApp
    "${CMAKE_SOURCE_DIR}/main.cu"
)

target_include_directories(
    3dPathTracerApp
    PUBLIC
        ${OPTIX_INCLUDE_DIR}
)


target_link_libraries(
    3dPathTracerApp 
    PRIVATE
        PATHTRACER::pathtracer
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
    PRIVATE
        gtest_main
        PATHTRACER::pathtracer
)

target_include_directories(
    3dPathTracerTests
    PRIVATE
        "${CMAKE_SOURCE_DIR}/tests"
)