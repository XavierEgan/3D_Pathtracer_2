# STOLEN FROM HERE: 
# https://github.com/NVIDIA/OptiX_Apps/blob/master/3rdparty/CMake/FindOptiX90.cmake
# i changed it a lil bit

# Looks for the environment variable:
# OPTIX90_PATH

# Sets the variables upon sucess:
# OPTIX90_PATH
# OPTIX90_INCLUDE_DIR
# OptiX90_FOUND

# Sets generic aliases
# OPTIX_PATH
# OPTIX_INCLUDE_DIR
# OPTIX_FOUND

set(OPTIX90_PATH $ENV{OPTIX90_PATH})

if ("${OPTIX90_PATH}" STREQUAL "")
	if (WIN32)
		# Try finding it inside the default installation directory under Windows first.
		set(OPTIX90_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
	else()
		# Adjust this if the OptiX SDK 9.0.0 installation is in a different location.
		set(OPTIX90_PATH "$ENV{HOME}/NVIDIA-OptiX-SDK-9.0.0-linux64")
	endif()
endif()

find_path(OPTIX90_INCLUDE_DIR 
	NAMES optix.h optix_host.h optix_stubs.h optix_types.h
	HINTS "${OPTIX90_PATH}/include"
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPTIX90 DEFAULT_MSG OPTIX90_INCLUDE_DIR)
mark_as_advanced(OPTIX90_INCLUDE_DIR)

message("OPTIX90_INCLUDE_DIR = " "${OPTIX90_INCLUDE_DIR}")
message("OPTIX90_FOUND = " "${OPTIX90_FOUND}")

set(OPTIX_PATH ${OPTIX90_PATH})
set(OPTIX_INCLUDE_DIR ${OPTIX90_INCLUDE_DIR})
set(OPTIX_FOUND ${OPTIX90_FOUND})