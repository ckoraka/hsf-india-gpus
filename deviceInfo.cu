#include <stdio.h>

int main()
{
 int nDevices;
 cudaGetDeviceCount(&nDevices);
 printf("--------------------------------------------------------------------------------------------------------------------------------\n");
 for (int i = 0; i < nDevices; i++)
 {
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, i);
   printf("Device Number: %d\n", i);
   printf("  Device name: %s\n", prop.name);
   printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
   printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
   printf("  Compute capability: %d.%d\n",prop.major,prop.minor);
   printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
   printf("  Number of SMs: %d\n", prop.multiProcessorCount);
   printf("  Maximum grid dimensions:  %d,%d,%d\n",prop.maxGridSize[0] ,prop.maxGridSize[1] ,prop.maxGridSize[2]);
   printf("  Warp size  %d\n",prop.warpSize);
   printf("  Max # of threads / block: %d\n",prop.maxThreadsPerBlock);
   printf("  Max size of a block blockDim.x : %d, .y : %d, .z : %d \n",prop.maxThreadsDim[0], prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
 }
 printf("--------------------------------------------------------------------------------------------------------------------------------\n");
}

