

#include <iostream>

#include <cuda/cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void vecAdd(float* a, float* b, float* c, int vectorSize)
{
	//What index should the threads be working on?
	int workindex = threadIdx.x + blockDim.x * blockIdx.x;

	if (workindex < vectorSize)
	{
		c[workindex] = a[workindex] + b[workindex];
	}
}

void initArray(float* a, int size)
{
	for (int i = 0; i < size; i++)
	{
		a[i] = rand() % (100 + 1 - 0);
	}
}

//Runs on the CPU
void HostVecAdd(float* a, float* b, float* c, int size)
{
	for (size_t i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

bool vecApproximatelyEqual(float* a, float* b, int size, float epsilon = 0.00001)
{
	for (size_t i = 0; i < size; i++)
	{
		if (fabs(a[i] - b[i]) > epsilon) {

			printf("Index %d mismatch: %f != %f\n", i, a[i], b[i]);
			return false;
		}
	}
	return true;
}

void unifiedMemExample(int vectorSize)
{

	//Pointers to memory vectors. Will run on the GPU
	float* a = nullptr;
	float* b = nullptr;
	float* c = nullptr;

	//Used on the CPU
	float* comparisonResult = (float*)malloc(vectorSize * sizeof(float));

	//Use unified memory to allocate buffers
	cudaMallocManaged(&a, vectorSize * sizeof(float));
	cudaMallocManaged(&b, vectorSize * sizeof(float));
	cudaMallocManaged(&c, vectorSize * sizeof(float));

	//init vectors on host
	initArray(a, vectorSize);
	initArray(b, vectorSize);

	//Launch kernel. Unified memory will make sure a, b and c are
	//accessible to the GPU
	int threads = 256;
	int blocks = cuda::ceil_div(vectorSize, threads);
	vecAdd<<<blocks, threads>>>(a, b, c, vectorSize);

	HostVecAdd(a, b, comparisonResult, vectorSize);

	//Compair the results on the CPU to the GPU
	if (vecApproximatelyEqual(c, comparisonResult, vectorSize))
	{
		printf("Unified Memory: CPU and GPU answers match\n");
	}
	else
	{
		printf("Unified Memory: Error - CPU and GPU answers do not match\n");
	}

	//Clean up
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	free(comparisonResult);

}

int main()
{
	unifiedMemExample(1024);
}