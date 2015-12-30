#include <stdio.h>

int main(int argc, char *argv[])
{
	// Get the number of devices.
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("deviceCount: %d\n", deviceCount);

	// Get the properties of device 0.
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	// Print some properties.
	// Refer to driver_types.h for a complete list of properties.
	printf("name: %s\n", deviceProp.name);
	printf("major: %d\n", deviceProp.major);
	printf("minor: %d\n", deviceProp.minor);
	printf("multiProcessorCount: %d\n", deviceProp.multiProcessorCount);
	printf("totalGlobalMem: %d B = %d MB\n", deviceProp.totalGlobalMem, deviceProp.totalGlobalMem / 1048576);
	printf("sharedMemPerBlock: %d B = %d KB\n", deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock / 1024);
	printf("totalConstMem: %d B = %d KB\n", deviceProp.totalConstMem, deviceProp.totalConstMem / 1024);
	printf("regsPerBlock: %d\n", deviceProp.regsPerBlock);
	printf("ECCEnabled: %d\n", deviceProp.ECCEnabled);
	printf("kernelExecTimeoutEnabled: %d\n", deviceProp.kernelExecTimeoutEnabled);
	printf("clockRate: %d KHz = %d MHz\n", deviceProp.clockRate, deviceProp.clockRate / 1000);
	printf("memoryClockRate: %d KHz = %d MHz\n", deviceProp.memoryClockRate, deviceProp.memoryClockRate / 1000);
	printf("memoryBusWidth: %d bits\n", deviceProp.memoryBusWidth);
	printf("l2CacheSize: %d B = %d KB\n", deviceProp.l2CacheSize, deviceProp.l2CacheSize / 1024);
	printf("warpSize: %d\n", deviceProp.warpSize);
	printf("maxThreadsPerMultiProcessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("maxThreadsPerBlock: %d\n", deviceProp.maxThreadsPerBlock);
	printf("maxThreadsDim[0]: %d\n", deviceProp.maxThreadsDim[0]);
	printf("maxThreadsDim[1]: %d\n", deviceProp.maxThreadsDim[1]);
	printf("maxThreadsDim[2]: %d\n", deviceProp.maxThreadsDim[2]);
	printf("maxGridSize[0]: %d\n", deviceProp.maxGridSize[0]);
	printf("maxGridSize[1]: %d\n", deviceProp.maxGridSize[1]);
	printf("maxGridSize[2]: %d\n", deviceProp.maxGridSize[2]);
	printf("deviceOverlap: %d\n", deviceProp.deviceOverlap);
	printf("asyncEngineCount: %d\n", deviceProp.asyncEngineCount);
	printf("integrated: %d\n", deviceProp.integrated);
	printf("canMapHostMemory: %d\n", deviceProp.canMapHostMemory);
	printf("concurrentKernels: %d\n", deviceProp.concurrentKernels);
	printf("tccDriver: %d\n", deviceProp.tccDriver);
	printf("unifiedAddressing: %d\n", deviceProp.unifiedAddressing);
	printf("pciBusID: %d\n", deviceProp.pciBusID);
	printf("pciDeviceID: %d\n", deviceProp.pciDeviceID);
	printf("computeMode: %d\n", deviceProp.computeMode);
	if (deviceProp.computeMode == cudaComputeModeDefault) printf("computeMode: %s\n", "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)");
	if (deviceProp.computeMode == cudaComputeModeExclusive) printf("computeMode: %s\n", "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)");
	if (deviceProp.computeMode == cudaComputeModeProhibited) printf("computeMode: %s\n", "Prohibited (no host thread can use ::cudaSetDevice() with this device)");
	if (deviceProp.computeMode == cudaComputeModeExclusiveProcess) printf("computeMode: %s\n", "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)");
}

