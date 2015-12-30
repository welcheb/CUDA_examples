#include <stdio.h>

__global__ void helloFromGPU(void)
{
	printf("Hello World from GPU! threadIdx.x=%d\n", threadIdx.x);
}

int main(void)
{
	// hello from CPU
	printf("Hello World from CPU!\n");

	// hello from GPU
	helloFromGPU <<<1, 10>>>();
	cudaDeviceReset();
	return(0);
}



