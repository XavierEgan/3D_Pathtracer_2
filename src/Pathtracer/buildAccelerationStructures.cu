#include "CommonInclude.hpp"
#include "Mesh.hpp"
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <vector>

pt::PtError buildAccelerationStructures(std::vector<pt::Mesh> meshs) {
	OptixAccelBuildOptions accelOptions = {};
	OptixBuildInput buildInputs[2];

	CUdeviceptr tempBuffer, outputBuffer;
	size_t tempBufferSizeInBytes, outputBufferSizeInBytes;

	memset(&accelOptions, 0, sizeof(OptixAccelBuildOptions));
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accelOptions.motionOptions = BUILD_OPERATION_


	return pt::PtErrorType::OK;
}