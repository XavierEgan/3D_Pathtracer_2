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

# find packages
find_package(CUDAToolkit REQUIRED) # set CUDA_INCLUDE_DIRS, CUDA::cudart, CUDA::cuda_driver
find_package(OPTIX90 REQUIRED)     # gets OptiX_INCLUDE

# make sure we have optix
if(NOT OPTIX_INCLUDE_DIR)
    message(FATAL_ERROR "OPTIX_INCLUDE_DIR must be set")
endif()

# get pt sources excluding .optix ones
file(GLOB_RECURSE PATH_TRACER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cu"
)

# get optix kernel sources
file(GLOB_RECURSE OPTIX_DEVICE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.optix.cu"
)

# remove .optix.cu files from PATH_TRACER_SOURCES
list(REMOVE_ITEM PATH_TRACER_SOURCES ${OPTIX_DEVICE_SOURCES})

foreach(file ${OPTIX_DEVICE_SOURCES})
    message(STATUS "optix file found: ${file}")
endforeach()

# make OPTIX_IR_OUTPUTS variable
set(OPTIX_IR_OUTPUTS "")

# loop over each optix file and compile it
foreach(src ${OPTIX_DEVICE_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    set(out ${CMAKE_CURRENT_BINARY_DIR}/${name}.optixir)

    add_custom_command(
        OUTPUT ${out}
        COMMAND nvcc
            -optix-ir
            --use_fast_math
            -I${OPTIX_INCLUDE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            ${src}
            -o ${out}
        DEPENDS ${src}
        COMMENT "Building OptiX-IR: ${name}.optixir"
        VERBATIM
    )

    list(APPEND OPTIX_IR_OUTPUTS ${out})
endforeach()

foreach(file ${OPTIX_IR_OUTPUTS})
    message(STATUS "optix file compiled: ${file}")
endforeach()

# makes a buidable target (cmake --build . --target optix_ir)
add_custom_target(optix_ir ALL DEPENDS ${OPTIX_IR_OUTPUTS})

# make library to link to
add_library(PATHTRACER ${PATH_TRACER_SOURCES})

# make it so that optix_ir is compiled with PATHTRACER
add_dependencies(PATHTRACER optix_ir)

# find the include dir of the pathtracer without hardcoding it
find_path(PATHTRACER_INCLUDE_DIR
    NAMES
        Conversions.hpp
        ObjParser.hpp
    HINTS
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/include" "$ENV{PATHTRACER_ROOT}"
)

# make sure we found it
if(NOT PATHTRACER_INCLUDE_DIR)
    message(FATAL_ERROR "PATHTRACER_INCLUDE_DIR not found")
endif()

# link cuda runtime and cuda driver to PATHTRACER
target_link_libraries(PATHTRACER
    PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
)

# set the include directories
target_include_directories(PATHTRACER
    PRIVATE
        ${PATHTRACER_INCLUDE_DIR}
        ${OPTIX_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_library(PATHTRACER::pathtracer ALIAS PATHTRACER)