#include <stdio.h>
#include <time.h>

//Set 2-D Matrix size: SIZE*SIZE
const int DSIZE = 256;
const int a = 1;
const int b = 2;

// error checking macro
#define cudaCheckErrors()                                       \
	do {                                                        \
		cudaError_t __err = cudaGetLastError();                 \
		if (__err != cudaSuccess) {                             \
			fprintf(stderr, "Error:  %s at %s:%d \n",           \
			cudaGetErrorString(__err),__FILE__, __LINE__);      \
			fprintf(stderr, "*** FAILED - ABORTING***\n");      \
			exit(1);                                            \
		}                                                       \
	} while (0)

// Check if matrix multiplication was correct
int check_result(const int *C){

	for (int i = 0; i < DSIZE*DSIZE; i++) {
		if (C[i] != a*b*DSIZE) {
			printf("Error : Index %d was %d instead of %d\n", i, C[i], a*b*DSIZE);
			return -1;
		}
	}
	printf("Matrix multiplication was correct!\n");
	return 0;
}

// Function that runs on the CPU
void matrix_mult_cpu(const int *A, const int *B, int *C, int N) {

	for(int i=0; i<N; ++i)
	{
		for(int j=0; j<N; ++j)
		{
			int sum = 0; 
			for(int k=0; k<N; ++k)
				sum += A[k+N*i] * B[j+k*N];
			C[j+N*i] = sum;
		}
	}
}

// CUDA kernel that runs on the GPU
__global__ void matrix_mult_gpu(const int *A, const int *B, int *C, int N) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

	if(idx<N && idy<N){
		int sum = 0; 
		for(int k=0; k<N; k++)
			sum += A[k+N*idx] * B[idy+k*N];
		C[idy+N*idx] = sum;
	}
}

int main() {

	// Variables used to measure time
	clock_t t0, t1, t2, t3;
	double t_cpu = 0.0;
	double t_gpu = 0.0;

	// Create the device and host pointers
	int *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

	// Fill in the host pointers 
	h_A = new int[DSIZE*DSIZE];
	h_B = new int[DSIZE*DSIZE];
	h_C = new int[DSIZE*DSIZE];
	for (int i = 0; i < DSIZE*DSIZE; i++){
		h_A[i] = a;
		h_B[i] = b;
		h_C[i] = 0;
	}

	// Measure time on CPU : Start timing
	t0 = clock();

	// Call the CPU function
	matrix_mult_cpu(h_A,h_B,h_C,DSIZE);

	// Calculate & print CPU time
	t1 = clock();
	t_cpu = ((double)(t1-t0))/CLOCKS_PER_SEC;
	printf ("CPU took %f seconds\n", t_cpu);

	// Check if matrix multiplication on CPU was correct
	check_result(h_C);

	// Initialize host pointer that holds result
	for (int i = 0; i < DSIZE*DSIZE; i++)
		h_C[i] = 0;


	// Measure time on GPU : Start timing
	t2 = clock();

	// Allocate device memory 
	cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(int));
	cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(int));
	cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(int));
	// Check memory allocation for errors
	cudaCheckErrors();

	// Copy the matrices on GPU
	cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(int), cudaMemcpyHostToDevice);
	// Check memory copy for errors
	cudaCheckErrors();

	// Define the number of threads per block
	int blockDim = 32;
	// dim3: Native CUDA type used to specify dimensions (up to 3 arguments)
	dim3 block(blockDim, blockDim);
	// Define the number of blocks in the grid
	dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
	// Launch kernel
	matrix_mult_gpu<<<grid, block>>>(d_A, d_B, d_C, DSIZE);

	// Check kernel launch for errors
	cudaCheckErrors();

	// Copy results back to host
	cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(int), cudaMemcpyDeviceToHost);

	// Measure time on GPU
	t3 = clock();
	t_gpu = ((double)(t3-t2))/CLOCKS_PER_SEC;
	printf ("GPU took %f seconds\n", t_gpu);

	// Check if matrix multiplication on GPU was correct
	check_result(h_C);

	// Free the allocated memory 
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(h_A);
	cudaFree(h_B);
	cudaFree(h_C);

	return 0;

}
