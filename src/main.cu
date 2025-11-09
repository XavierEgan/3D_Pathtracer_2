#include <iostream>
#include <cuda_runtime.h>
#include <optix_stubs.h>

int main(void) {
	// initialize optix
	optixInit();

	// make the cuda context
	OptixDeviceContext context = nullptr;
	cudaFree(0);
	CUcontext cuCtx = 0;
	optixDeviceContextCreate(cuCtx, 0, &context);

	
}