#include <cuda/cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void vecAdd(float* a, float* b, float* c)
{
	int workindex = threadIdx.x + blockDim.x * blockIdx.x;

	c[workindex] = a[workindex] + b[workindex];
}

int main()
{

	int thread = 256;
	int blocks = cuda::ceil_div(4, thread);

	float a[4] = { 1,2,3,4 };
	float b[4] = { 1,2,3,4 };
	float c[4] = {};
	vecAdd << <blocks, thread >> > (a, b, c);

}