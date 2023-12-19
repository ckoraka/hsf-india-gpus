// dot_product.cu
// nvcc dot_product.cu -o dot_product

#include <stdio.h>
#include <time.h>


#define BLOCK_SIZE 32

const int DSIZE = 256;
const int a = 1;
const int b = 1;

// error checking macro
#define cudaCheckErrors(msg)                                    \
	do {                                                        \
		cudaError_t __err = cudaGetLastError();                 \
		if (__err != cudaSuccess) {                             \
			fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
					msg, cudaGetErrorString(__err),             \
					__FILE__, __LINE__);                        \
			fprintf(stderr, "*** FAILED - ABORTING***\n");      \
			exit(1);                                            \
		}                                                       \
	} while (0)


// CUDA kernel that runs on the GPU
__global__ void dot_product(const int *A, const int *B, int *C, int N) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N) {
		printf("Adding %i \n", A[idx] * B[idx]);
		atomicAdd(C, A[idx] * B[idx]);
	}

}


int main() {

	// Create the device and host pointers
	int *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

	// Fill in the host pointers 
	h_A = new int[DSIZE];
	h_B = new int[DSIZE];
	h_C = new int;
	for (int i = 0; i < DSIZE; i++){
		h_A[i] = a;
		h_B[i] = b;
	}

	*h_C = 0;

	// Allocate device memory 
	cudaMalloc(&d_A, DSIZE*sizeof(int));
	cudaMalloc(&d_B, DSIZE*sizeof(int));
	cudaMalloc(&d_C, sizeof(int));

	// Check memory allocation for errors
	cudaCheckErrors("Was memory allocation successful");

	// Copy the vectors on GPU
	cudaMemcpy(d_A, h_A, DSIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, DSIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, sizeof(int),       cudaMemcpyHostToDevice);

	// Check memory copy for errors
	cudaCheckErrors("Was memory copy from host to device successful");

	// Define block/grid dimensions and launch kernel
	int grid_size = (DSIZE/BLOCK_SIZE);
	// myKernel<<<nBlocks, nThreads>>>
	dot_product<<<grid_size, BLOCK_SIZE>>>(d_A, d_B, d_C, DSIZE);

	// Copy results back to host
	cudaMemcpy(h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost);

	// Check copy for errors
	cudaCheckErrors("Was memory copy from device to host successful");

	// Verify result
	printf("After kernel: value of h_C is %i\n", *h_C);

	// Free allocated memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
