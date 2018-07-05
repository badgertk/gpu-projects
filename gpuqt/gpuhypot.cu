#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <sys/time.h>

void cpuhypot(int *CPU_idata, int *CPU_odata, size_t Totsize);
cudaError_t gpuhypot(int *CPU_idata, int *CPU_odata, size_t Totsize, float* Runtimes);

#define BLOCK_SIZE 128
#define NUM_BLOCKS 32768
const int arraySize = NUM_BLOCKS*BLOCK_SIZE;  // Must be a power 2 times BLOCK_SIZE (this code cannot handle other cases)

int *CPU_InputArray;
int *CPU_OutputArray;

__global__ void hypotKernelG(int *GPU_i, int *GPU_o)
{
  // unsigned int  tid = threadIdx.x;                        // gets index of thread in block
  unsigned int  i = blockIdx.x*blockDim.x + threadIdx.x;  // Gets the global position of this thread

  float a,b,x;
  int II;
  
  II = i << 1;
  a = (float)GPU_i[II];
  b = (float)GPU_i[II+1];
  x = sqrt( a*a + b*b );
  GPU_o[i] = (int)x;  // Note: CPU/GPU results may be different in the case of overflow.
}

int main()
{
  float GPURuntimes[4];         // Run times of the GPU code
  struct timeval st, et;
  double StartTime, EndTime;
  cudaError_t cudaStatus;
  
  // Create CPU memory to store the input and output arrays
  CPU_InputArray  = (int*)malloc(arraySize*sizeof(int));
  if(CPU_InputArray == NULL) {
    fprintf(stderr,"OOPS. Can't create InputArray using malloc() ...\n\n");
    return EXIT_FAILURE;
  }
  CPU_OutputArray = (int*)malloc(arraySize*sizeof(int)/2);
  if(CPU_OutputArray == NULL) {
    free(CPU_InputArray);
    fprintf(stderr,"OOPS. Can't create OutputArray using malloc() ...\n\n");
    return EXIT_FAILURE;
  }
  
  for (int i = 0; i < arraySize; i++)	{
    CPU_InputArray[i] = (int) rand()*10000000;  // create random numbers from 0 to some big value
  }
  
  // Run it in the CPU to get the gold copy
  gettimeofday(&st, NULL);
  cpuhypot(CPU_InputArray, CPU_OutputArray, arraySize/2);
  gettimeofday(&et, NULL);
  StartTime = st.tv_sec*1000.00 + (st.tv_usec/1000.0);
  EndTime   = et.tv_sec*1000.00 + (et.tv_usec/1000.0);
  printf("Elapsed time = %ld ms\n", (long) (EndTime - StartTime));

  // Compute hypot in parallel.
  cudaStatus = gpuhypot(CPU_InputArray, CPU_OutputArray, arraySize, GPURuntimes);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "\n gpuhypot failed!\n");
    free(CPU_InputArray);
    free(CPU_OutputArray);
    cudaDeviceReset();
    return EXIT_FAILURE;
  }

  printf("\nKERNEL = hypotKernelG ...\n");
  printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \n Total=%5.2f ms\n",
	 GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
  printf("-----------------------------------------------------------------\n");

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    free(CPU_InputArray);
    free(CPU_OutputArray);
    return EXIT_FAILURE;
  }
  
  free(CPU_InputArray);
  free(CPU_OutputArray);
  return EXIT_SUCCESS;
}

void cpuhypot(int *CPU_idata, int *CPU_odata, size_t Totsize)
{
  int n;
  float a,b,x;
  int II;
  
  for(n=0; n<Totsize; n++){
    II = n << 1;
    a = (float)CPU_idata[II];
    b = (float)CPU_idata[II+1];
    x = sqrtf(a*a + b*b);
    CPU_odata[n] = (int) x;
  }
}

// Helper function for using CUDA to compute hypot in parallel.
cudaError_t gpuhypot(int *CPU_idata, int *CPU_odata, size_t Totsize, float* Runtimes)
{
  cudaEvent_t time1, time2, time3, time4;
  int TotalGPUSize;
  
  int *GPU_idata = 0;
  int *GPU_odata = 0;
  //int blocksize;
  //int totalBlocks = Totsize/BLOCK_SIZE;
  
  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaError_t cudaStatus;
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    goto Error;
  }
  
  cudaEventCreate(&time1);
  cudaEventCreate(&time2);
  cudaEventCreate(&time3);
  cudaEventCreate(&time4);
  
  cudaEventRecord(time1, 0);
  // Allocate GPU buffer for inputs and outputs (hypotenuse)
  TotalGPUSize=Totsize *sizeof(int);
  
  cudaStatus = cudaMalloc((void**)&GPU_idata, TotalGPUSize);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  
  cudaStatus = cudaMalloc((void**)&GPU_odata, TotalGPUSize/2);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  
  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(GPU_idata, CPU_idata, TotalGPUSize, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }
  
  cudaEventRecord(time2, 0);
  // Launch a kernel on the GPU with one thread for each element.
  
  hypotKernelG<<<NUM_BLOCKS/2, BLOCK_SIZE>>>(GPU_idata, GPU_odata);

  // Check for errors immediately after kernel launch.
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "error code %d (%s) launching kernel!\n",
	      cudaStatus, cudaGetErrorString(cudaStatus));
      goto Error;
    }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching kernel!\n",
	    cudaStatus, cudaGetErrorString(cudaStatus));
    goto Error;
  }
  
  cudaEventRecord(time3, 0);
  // Copy output (results) from GPU buffer to host (CPU) memory.
  cudaStatus = cudaMemcpy(CPU_odata, GPU_odata, TotalGPUSize/2, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }
  
  cudaEventRecord(time4, 0);
  cudaEventSynchronize(time1);
  cudaEventSynchronize(time2);
  cudaEventSynchronize(time3);
  cudaEventSynchronize(time4);
  
  float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime;
  
  cudaEventElapsedTime(&totalTime, time1, time4);
  cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
  cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
  cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);
  
  Runtimes[0] = totalTime;
  Runtimes[1] = tfrCPUtoGPU;
  Runtimes[2] = kernelExecutionTime;
  Runtimes[3] = tfrGPUtoCPU;

 Error:
  cudaFree(GPU_odata);
  cudaFree(GPU_idata);
  cudaEventDestroy(time1);
  cudaEventDestroy(time2);
  cudaEventDestroy(time3);
  cudaEventDestroy(time4);
  
  return cudaStatus;
}
