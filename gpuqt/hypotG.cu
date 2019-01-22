
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <Windows.h>


cudaError_t gpuhypot(int *CPU_idata, int *CPU_odata, size_t Totsize, float* Runtimes);

#define BLOCK_SIZE	 256

const int arraySize = 32768*BLOCK_SIZE; //Must be a power 2 times BLOCK_SIZE (this code cannot handle other cases)

int *CPU_InputArray;
int *CPU_OutputArray;



__global__ void hypotKernelG(int *GPU_i, int *GPU_o)
{
	unsigned int  tid = threadIdx.x;			// gets index of thread in block
	unsigned int  bid = blockIdx.x*blockDim.x;  // gets index of the block
	unsigned int  i = bid+tid;					// global index of this thread
	int           a,b;                          // temp variables for this thread to use

	a = GPU_i[2*i];
	b = GPU_i[2*i+1];
    GPU_o[i] = sqrt( (double) (a*a + b*b) );
}



int main()
{
	float GPURuntimes[4];         // Run times of the GPU code
	clock_t CPUStartTime, CPUEndTime, CPUElapsedTime;
	cudaError_t cudaStatus;
	int InputArraySize,OutputArraySize;
	char key;

	// Create CPU memory to store the input and output arrays
	InputArraySize=arraySize*sizeof(int);
	OutputArraySize=arraySize*sizeof(int)/2;
	CPU_InputArray  = (int*)malloc(InputArraySize);
	if(CPU_InputArray == NULL){ fprintf(stderr,"OOPS. Can't create InputArray using malloc() ...\n\n"); exit(EXIT_FAILURE); }
	CPU_OutputArray = (int*)malloc(OutputArraySize);
	if(CPU_OutputArray == NULL){ fprintf(stderr,"OOPS. Can't create OutputArray using malloc() ...\n\n"); exit(EXIT_FAILURE); }

	for (int i = 0; i < arraySize; i++)	{
		CPU_InputArray[i] = (int) rand()*10000000;       // create random numbers from 0 to some big value
	}

	cudaStatus = gpuhypot(CPU_InputArray, CPU_OutputArray, arraySize, GPURuntimes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n gpuhypot failed!");
		key=getc(stdin);
		free(CPU_InputArray);
		free(CPU_OutputArray);
		return 1;
	}
	printf("\nKERNEL = hypotKernelG ...\n");
	printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \n Total=%5.2f ms\n\n",GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
	printf("Tfr CPU->GPU = %5.2f ms  ...  %6d MB  ...  %6.3f GB/s\n",GPURuntimes[1],InputArraySize/1024/1024,(float)InputArraySize/(GPURuntimes[1]*1024.0*1024.0));
	printf("Tfr GPU->CPU = %5.2f ms  ...  %6d MB  ...  %6.3f GB/s\n",GPURuntimes[3],OutputArraySize/1024/1024,(float)OutputArraySize/(GPURuntimes[3]*1024.0*1024.0));
	printf("--------------------------------------------------------------------\n");
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		key=getc(stdin);
	    free(CPU_InputArray);
	    free(CPU_OutputArray);
        return 1;
    }

	key=getc(stdin);   // wait for a char, so the terminal window doesn't close

    free(CPU_InputArray);
	free(CPU_OutputArray);
    return 0;
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t gpuhypot(int *CPU_idata, int *CPU_odata, size_t Totsize, float* Runtimes)
{
	cudaEvent_t time1, time2, time3, time4;
	int TotalGPUSize;

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

    int *GPU_idata = 0;
    int *GPU_odata = 0;
    

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
	int blocksize = BLOCK_SIZE;
	int tempSize = Totsize;
	int totalBlocks = tempSize/BLOCK_SIZE;
	int *tempOut;
	
	hypotKernelG<<<16384,256>>>(GPU_idata, GPU_odata);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
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