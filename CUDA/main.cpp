#include <cuda/cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

//Runs on the GPU
__device__ void vecAdd(float* a, float* b, float* c, int vectorSize)
{
	//What index should the threads be working on?
	int workindex = threadIdx.x + blockDim.x * blockIdx.x;

	if (workindex < vectorSize)
	{
		c[workindex] = a[workindex] + b[workindex];
	}
}

//Runs on the CPU
__host__ void VecAdd(float* a, float* b, float* c, int vectorSize)
{
	int thread = 256;
	int blocks = cuda::ceil_div(4, thread);

	vecAdd<<<blocks, thread >>>(a, b, c);
}

int main()
{

	float a[4] = { 1,2,3,4 };
	float b[4] = { 1,2,3,4 };
	float c[4] = {0,0,0,0};

	//Wrapper that calls a cuda kernel
	VecAdd(a, b, c, 4);

	for (auto i : c) {
		std::cout << i << std::endl;
	}
}