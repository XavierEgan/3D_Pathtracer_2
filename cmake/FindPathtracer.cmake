# sets PATHTRACER_INCLUDE_DIR and PATHTRACER_LIB

# requires OPTIX_INCLUDE_DIR to already be set

# Sets the variables upon sucess:
# PATHTRACER_PATH
# PATHTRACER_INCLUDE_DIR
# PATHTRACER_FOUND

# Sets generic aliases
# OPTIX_PATH
# OPTIX_INCLUDE_DIR
# OPTIX_FOUND

if(NOT OPTIX_INCLUDE_DIR)
    message(FATAL_ERROR "OPTIX_INCLUDE_DIR must be set")
endif()

file(GLOB_RECURSE PATH_TRACER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cu"
)

add_library(PATHTRACER ${PATH_TRACER_SOURCES})

find_path(PATHTRACER_INCLUDE_DIR
    NAMES
        Conversions.hpp
        ObjParser.hpp
    HINTS
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/include"
        "$ENV{PATHTRACER_ROOT}"
    PATH_SUFFIXES
        include
)

if(NOT PATHTRACER_INCLUDE_DIR)
    message(FATAL_ERROR "PATHTRACER_INCLUDE_DIR not found")
endif()

target_link_libraries(PATHTRACER
    PRIVATE
        CUDA::cudart
)

target_include_directories(PATHTRACER
    PRIVATE
        ${PATHTRACER_INCLUDE_DIR}
        ${OPTIX_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_library(PATHTRACER::pathtracer ALIAS PATHTRACER)