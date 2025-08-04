#include "optix.h"
#include "optix_stubs.h"
#include "cuda_runtime.h"
#include "glm/glm.hpp"

int main() {
	// make an Optix context to manage the gpu (section 4)
	OptixDeviceContext optixContext = nullptr;
	cudaFree(0);
	CUcontext cuCtx = 0;
	optixDeviceContextCreate(cuCtx, 0, &optixContext);

	// create a cuStream (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
	CUstream cuStream = nullptr;
	cuStreamCreate(&cuStream, CU_STREAM_NON_BLOCKING);

	// disable caching (section 4.2)
	optixDeviceContextSetCacheEnabled(optixContext, 0);

	/*------------------------------------------------------------*/

	// build an AS (section 5)
	OptixAccelBuildOptions accelOptions = {};
	OptixBuildInput buildInputs[1];

	memset((void *) buildInputs, 0, sizeof(OptixBuildInput) * 2);

	CUdeviceptr tempBuffer, outputBuffer;
	size_t tempBufferSizeInBytes, outputBufferSizeInBytes;

	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	accelOptions.motionOptions.numKeys = 0;

	// make the traingle array
	buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	OptixBuildInput& buildInput = buildInputs[0];

	int numVertices = 3;
	CUdeviceptr d_vertexBuffer;

	int numTriangles = 1;
	CUdeviceptr d_indexBuffer;

	cudaMalloc((void **) &d_vertexBuffer, sizeof(float3) * 3);
	cudaMalloc((void **) &d_indexBuffer, sizeof(int) * 3);

	float3 verts[3] = {
		make_float3(-1,  1, 1),
		make_float3(-1, -1, 1),
		make_float3( 1, -1, 1),
	};

	int indices[3] = {
		0, 1, 2
	};

	cudaMemcpy((void *) d_vertexBuffer, verts, sizeof(float3) * 3, cudaMemcpyHostToDevice);
	cudaMemcpy((void *) d_indexBuffer, indices, sizeof(int) * 3, cudaMemcpyHostToDevice);

	buildInput.triangleArray.vertexBuffers = &d_vertexBuffer;
	buildInput.triangleArray.numVertices = numVertices;
	buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	buildInput.triangleArray.vertexStrideInBytes = 0;
	buildInput.triangleArray.indexBuffer = d_indexBuffer;
	buildInput.triangleArray.numIndexTriplets = numTriangles;
	buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	buildInput.triangleArray.indexStrideInBytes = 0;
	buildInput.triangleArray.preTransform = 0;
	
	OptixAccelBufferSizes bufferSizes = {};
	optixAccelComputeMemoryUsage(optixContext, &accelOptions, buildInputs, 2, &bufferSizes);

	void* d_output;
	void* d_temp;

	cudaMalloc(&d_output, bufferSizes.outputSizeInBytes);
	cudaMalloc(&d_temp, bufferSizes.tempSizeInBytes);

	OptixTraversableHandle outputHandle = 0;
	OptixResult results = optixAccelBuild(
		optixContext, 
		cuStream,
		&accelOptions, 
		buildInputs, 
		2, 
		(CUdeviceptr) d_temp,
		bufferSizes.tempSizeInBytes, 
		(CUdeviceptr) d_output,
		bufferSizes.outputSizeInBytes, 
		&outputHandle, 
		nullptr, 
		0
	);

	cudaFree((void *) d_vertexBuffer);
	cudaFree((void *) d_indexBuffer);

	/*------------------------------------------------------------*/



	/*------------------------------------------------------------*/

	// destroy the context (section 4)
	optixDeviceContextDestroy(optixContext);

	// destroy the cuStream
	cuStreamDestroy(cuStream);
}