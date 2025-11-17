#include "render.hpp"

#include <optix_stubs.h>
#include <cuda_runtime.h>

#include "getIrData.hpp"

/*
 * Takes a vector of Mesh and Material
 * Then renders an image with Optix and returns it as a char*
 */
char* render(std::vector<pt::Mesh> meshs, std::vector<pt::Material> materials) {
	// initialize optix
	optixInit();

	// make the cuda context
	OptixDeviceContext optixContext = nullptr;
	cudaFree(0);
	CUcontext cuCtx = 0;
	optixDeviceContextCreate(cuCtx, 0, &optixContext);

	// create a cuStream (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
	CUstream cuStream = nullptr;
	cuStreamCreate(&cuStream, CU_STREAM_NON_BLOCKING);
	
/*****************************************************************************/
	// Build acceleration structures
/*****************************************************************************/
	
	// declare stuff
	OptixAccelBuildOptions accelOptions = {};
	memset(&accelOptions, 0, sizeof(OptixAccelBuildOptions));
	int numBuildInputs = meshs.size();
	std::unique_ptr<OptixBuildInput[]> buildInputs = std::make_unique<OptixBuildInput[]>(numBuildInputs);
	memset(buildInputs.get(), 0, sizeof(OptixBuildInput) * numBuildInputs);
	CUdeviceptr tempBuffer, outputBuffer;

	// set flags
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	accelOptions.motionOptions.numKeys = 0;

	// because the vert pointers need to live outside the loop, one per mesh/build input
	// (since they are a host allocated array of size one (pointer to pointer))
	std::unique_ptr<CUdeviceptr[]> vertPointers = std::make_unique<CUdeviceptr[]>(numBuildInputs);
	
	// flags are all the same and need to live outside loop
	unsigned int flags[1];
	flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

	// set up the build inputs (optix dev guide 9.0 was little help here because the provided code example is wrong lmao)
	for (int meshIndex = 0; meshIndex < numBuildInputs; meshIndex++) {
		const pt::Mesh& mesh = meshs.at(meshIndex);

		// make a build input
		OptixBuildInput buildInput = OptixBuildInput();
		buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// get reference to the tri array (its part of a union)
		OptixBuildInputTriangleArray& triangleArray = buildInput.triangleArray;

		int numVertices = mesh.vertices.size();

		CUdeviceptr& d_vertexBuffer = vertPointers[meshIndex];
		std::unique_ptr<float3[]> verts = std::make_unique<float3[]>(numVertices);

		// we need to turn the vertices into float 3
		// (i can probably just use vertices.data(), but to be safe ill convert them all
		for (int i = 0; i < numVertices; i++) {
			const glm::vec3& v = mesh.vertices.at(i);
			verts[i] = make_float3(v.x, v.y, v.z);
		}

		cudaMalloc(reinterpret_cast<void**>( & d_vertexBuffer), numVertices * sizeof(float3));
		cudaMemcpy(reinterpret_cast<void*>(d_vertexBuffer), verts.get(), numVertices * sizeof(float3), cudaMemcpyHostToDevice);

		triangleArray.vertexBuffers = &d_vertexBuffer;
		triangleArray.numVertices = numVertices;
		triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleArray.vertexStrideInBytes = sizeof(float3);

		// get numTrianges
		int numIndices = mesh.vertexIndexes.size();
		int numTriangles = numIndices / 3;

		CUdeviceptr d_indexBuffer;
		std::unique_ptr<uint3[]> indices = std::make_unique<uint3[]>(numTriangles);

		for (int i = 0; i < numTriangles; i++) {
			indices[i] = make_uint3(
				mesh.vertexIndexes.at(i * 3 + 0),
				mesh.vertexIndexes.at(i * 3 + 1),
				mesh.vertexIndexes.at(i * 3 + 2)
			);
		}
		
		cudaMalloc(reinterpret_cast<void**>(&d_indexBuffer), numTriangles * sizeof(int3));
		cudaMemcpy(reinterpret_cast<void*>(d_indexBuffer), indices.get(), numTriangles * sizeof(int3), cudaMemcpyHostToDevice);

		triangleArray.indexBuffer = d_indexBuffer;
		triangleArray.numIndexTriplets = numTriangles;
		triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleArray.indexStrideInBytes = sizeof(int3);
		triangleArray.preTransform = 0;

		// SBT 
		triangleArray.numSbtRecords = 1;
		triangleArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(nullptr);
		triangleArray.sbtIndexOffsetSizeInBytes = sizeof(int);
		triangleArray.sbtIndexOffsetStrideInBytes = sizeof(int);

		// SBT Flags
		triangleArray.flags = flags;

		buildInputs[meshIndex] = buildInput;
	}
	
	// calc memory usage
	OptixAccelBufferSizes bufferSizes = {};
	optixAccelComputeMemoryUsage(optixContext, &accelOptions, buildInputs.get(), numBuildInputs, &bufferSizes);

	// declare stuff
	CUdeviceptr d_output;
	CUdeviceptr d_temp;

	// malloc for optix to build in
	cudaMalloc((void**) &d_output, bufferSizes.outputSizeInBytes);
	cudaMalloc((void**) &d_temp, bufferSizes.tempSizeInBytes);

	// build the accel structure
	OptixTraversableHandle outputHandle = 0;
	OptixResult result = optixAccelBuild(optixContext, cuStream, &accelOptions, buildInputs.get(), numBuildInputs, d_temp, bufferSizes.tempSizeInBytes, d_output, bufferSizes.outputSizeInBytes, &outputHandle, nullptr, 0);
	
	if (result != OPTIX_SUCCESS) {
		// uh oh
		std::cerr << "optix Accel Build failed, error: " << result << std::endl;
	}

/*****************************************************************************/
	// Build the optix pipeline
/*****************************************************************************/

	OptixModuleCompileOptions moduleCompileOptions = {};
	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
	moduleCompileOptions.numPayloadTypes = 0;
	moduleCompileOptions.payloadTypes = 0;

	OptixPipelineCompileOptions pipelineCompileOptions = {};
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
	pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

	OptixModule module = nullptr;

	std::string irDataString = getIrData();

	const char* irData = irDataString.data();
	size_t irDataSize = irDataString.size();

	std::string logString;
	size_t logStringSize = sizeof(logString);

	OptixResult res = optixModuleCreate(optixContext, &moduleCompileOptions, &pipelineCompileOptions, irData, irDataSize, logString.data(), logStringSize, &module);

/*****************************************************************************/

/*****************************************************************************/

	return nullptr;
}