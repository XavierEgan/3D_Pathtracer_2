set(OptiX_ROOT_DIR $ENV{OptiX_INSTALL_DIR})

# Find the OptiX include directory
find_path(OptiX_INCLUDE_DIR
    NAMES optix.h
    PATHS ${OptiX_ROOT_DIR}/include
    DOC "OptiX include directory"
)

# Check if we found OptiX
if(OptiX_INCLUDE_DIR)
    set(OptiX_FOUND TRUE)
    message(STATUS "Found OptiX: ${OptiX_INCLUDE_DIR}")
else()
    set(OptiX_FOUND FALSE)
    message(FATAL_ERROR "OptiX not found at ${OptiX_ROOT_DIR}")
endif()