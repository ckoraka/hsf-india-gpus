#include <stdio.h>

__global__ void cuda_hello(){
	printf("Hello World from GPU. Running thread %d block  %d\n",threadIdx.x,blockIdx.x);
}

int main() {
	int gridDim = 1;
	int blockDim = 32;
	cuda_hello<<<gridDim,blockDim>>>();
	//FIXME:
	return 0;
}
