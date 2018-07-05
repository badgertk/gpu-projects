#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//CUDA STUFF:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//OpenCV stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes);

#define BOX_SIZE		1 		//ThreadsPerBlock == BOX_SIZE * BOX_SIZE
#define PI				3.1415926
#define EDGE			255
#define NOEDGE			0

int M; //number of rows in the image
int N; //number of columns in the image
int NumIter = 16;
int ThreshLo = 0;
Mat zero;

//ip.Vpixels <--> M
//ip.Hpixels <--> N

//kernels

int main(int argc, char *argv[]){
	float GPURuntimes[4]; //run times of the GPU code
	float ExecTotalTime, TfrCPUGPU, GPUTotalTime, TfrGPUCPU;
	cudaError_t cudaStatus;
	char filename[100]; //output file name
	int i;
	int *CPU_OutputArray = (int*) 0; //where the GPU should copy the output back to
	
	if (argc != 3){
		printf("Improper Usage!\n");
		printf("Usage: %s <input image> <output image>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	ExecTotalTime = 0;
	TfrCPUGPU = 0;
	GPUTotalTime = 0;
	TfrGPUCPU = 0;
	for (i = 0; i < NumIter; i++){

		//Load image:
		Mat image;
		image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
		if (! image.data){
			fprintf(stderr, "Could not open or find the image.\n");
			exit(EXIT_FAILURE);
		}
		printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.cols, image.rows, image.dims);
		
		//set up global variables for image size
		M = image.rows;
		N = image.cols;
		//Create CPU memory to store the output;
		zero = Mat(M,N,CV_8UC1, Scalar(255)); //start by making every pixel white
		sprintf(filename, "%s%d.png",argv[2],i);
		imwrite(filename, zero);
		
		CPU_OutputArray = (int*) malloc(M*N*sizeof(int));
		if (CPU_OutputArray == NULL){
			fprintf(stderr, "Oops, cannot create CPU_OutputArray using malloc() ...\n");
			exit(EXIT_FAILURE);
		}
		//run it
		cudaStatus = launch_helper(image, CPU_OutputArray, GPURuntimes);
		if (cudaStatus != cudaSuccess){
			fprintf(stderr, "launch_helper failed!\n");
			free(CPU_OutputArray);
			exit(EXIT_FAILURE);
		}
		// FIX THIS LAST
		printf("-----------------------------------------------------------------\n");
		printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \nSum of Iteration = %5.2f ms\n",
				GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
		ExecTotalTime += GPURuntimes[0];
		TfrCPUGPU += GPURuntimes[1];
		GPUTotalTime += GPURuntimes[2];
		TfrGPUCPU += GPURuntimes[3];
		printf("\nTotal Tfr CPU -> GPU Time = %5.2f ms\n", TfrCPUGPU);
		printf("GPU Execution Time = %5.2f ms \n", GPUTotalTime);
		printf("Total Tfr GPU -> CPU Time = %5.2f ms\n", TfrGPUCPU);
		printf("Total Execution Time = %5.2f ms\n", ExecTotalTime);
		printf("-----------------------------------------------------------------\n");
		
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess){
			fprintf(stderr, "cudaDeviceReset failed!\n");
			free(CPU_OutputArray);
			exit(EXIT_FAILURE);
		}

		//save image to disk
			Mat result = Mat(M,N,CV_8UC1, CPU_OutputArray);
			imwrite(filename,result);
		

		if (!imwrite(filename, result)){
			fprintf(stderr, "couldn't write output to disk!\n");
			free(CPU_OutputArray);
			exit(EXIT_FAILURE);
		}
		
		printf("Saved image '%s', size = %dx%d (dims = %d).\n",
			   //filename.c_str(), result.cols, result.rows, result.dims);
			   filename, result.cols, result.rows, result.dims);

		free(CPU_OutputArray);
	}
	exit(EXIT_SUCCESS);
		
}

cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes){
	
	cudaEvent_t time1, time2, time3, time4;
	int ucharGPUSize; // total size of 1 image in bytes
	uchar *GPU_idata;
	uchar *GPU_odata;
	uchar *GPU_zerodata;
	uchar *GPU_Dilationdata;
	uchar *GPU_Erosiondata;
	
	dim3 threadsPerBlock;
	dim3 numBlocks;
	
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0); // use the first GPU
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
	
	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

	cudaEventRecord(time1, 0);
	
	// Allocate GPU buffer for inputs and outputs:
	ucharGPUSize = M * N * sizeof(uchar);
	
	cudaStatus = cudaMalloc((void**)&GPU_idata, ucharGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_odata, ucharGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_zerodata, ucharGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_Dilationdata, ucharGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_Erosiondata, ucharGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPU_odata, zero.data, ucharGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyzero failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPU_idata, image.data, ucharGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaEventRecord(time2, 0);

	// Launch a kernel on the GPU with one thread for each pixel.
	threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
	numBlocks = dim3(M / threadsPerBlock.x, N / threadsPerBlock.y);

	//EROSION AND DILATION
	
	Dilation<<<numBlocks, threadsPerBlock>>>(GPU_idata, GPU_Dilationdata, M, N);

	// Check for errors immediately after kernel launch.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "error code %d (%s) launching kernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching addKernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}
	Erosion<<<numBlocks, threadsPerBlock>>>(GPU_idata, GPU_Erosiondata, M, N);

	// Check for errors immediately after kernel launch.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "error code %d (%s) launching kernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching addKernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	// THEN TAKE THE DIFFERENCE
	
	Difference<<<numBlocks, threadsPerBlock>>>(GPU_Dilationdata, GPU_Erosiondata, GPU_odata, M, N, ThreshLo*8, ThreshLo*8 + 7);

	// Check for errors immediately after kernel launch.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "error code %d (%s) launching kernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching addKernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	cudaEventRecord(time3, 0);
	// Copy output (results) from GPU buffer to host (CPU) memory.
	cudaStatus = cudaMemcpy(CPU_OutputArray, GPU_odata, ucharGPUSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
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
	cudaFree(GPU_zerodata);
	cudaFree(GPU_Dilationdata);
	cudaFree(GPU_Erosiondata);
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
	ThreshLo++;
	return cudaStatus;
}