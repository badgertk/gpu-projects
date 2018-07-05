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

#define BOX_SIZE		16 		//ThreadsPerBlock == BOX_SIZE * BOX_SIZE
#define PI				3.1415926
#define EDGE			255
#define NOEDGE			0

int M; //number of rows in the image
int N; //number of columns in the image
//int NumIter = 16;
Mat zero;

//ip.Vpixels <--> M
//ip.Hpixels <--> N




//kernels
__global__ void GaussianFilter(uchar *GPU_i, double *Gauss_o, int M, int N){

	int row = blockIdx.x * blockDim.x + threadIdx.x; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	//uchar output = GPU_i[idx];
	int i,j;
	double G;
	double Gauss[5][5] = {	{ 2, 4,  5,  4,  2 },
						{ 4, 9,  12, 9,  4 },
						{ 5, 12, 15, 12, 5 },
						{ 4, 9,  12, 9,  4 },
						{ 2, 4,  5,  4,  2 }	};
	
	if ((row < 2) || (row > (M - 3))) goto End;
	//col = 2;
	G = 0.0;
	int idx2, row2, col2;
	for (i=-2; i<=2; i++){
		for (j=-2; j<=2; j++){
			row2 = row + i;
			col2 = col + j;
			idx2 = row2*N + col2;
			G = G + GPU_i[idx2] * Gauss[i + 2][j + 2];	
		}
	}
	
	Gauss_o[idx] = G/ (double)159.00;	
	//printf("Gauss_o[] = %f", Gauss_o[idx]); //looks like the numbers are right
	End:;
}

__global__ void Sobel(double *Gauss_i, double *Gradient_o, double *Theta_o, int M, int N){
	int row = blockIdx.x * blockDim.x + threadIdx.x; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	//uchar output = Gauss_i[idx]; okay so these numbers are right
	//printf("Gauss_i[] = %f", Gauss_i[idx]);
	int i,j;
	double GX,GY;
	//printf("row = %d, col = %d", row, col);
	
double Gx[3][3] = {		{ -1, 0, 1 },
						{ -2, 0, 2 },
						{ -1, 0, 1 }	};

double Gy[3][3] = {		{ -1, -2, -1 },
						{  0,  0,  0 },
						{  1,  2,  1 }	};
	
	if ((row<1) || (row>(M-2))) goto End;
	//col = 1;
	if (col<=(N-2)){
		GX = 0.0; GY = 0.0;
		int row2, col2, idx2;
		for (i = -1; i <= 1; i++){
			for (j = -1; j<= 1; j++){
				row2 = row + i;
				col2 = col + j;
				//printf("row2 = %d, N = %d, col2 = %d", row2, N, col2);
				idx2 = row2*N + col2; //this is wrong
				GX = GX + Gauss_i[idx2] * Gx[i+1][j+1];
				//printf("Gauss_i[] = %f", Gauss_i[idx2]); //a lot of 124.92
				GY = GY + Gauss_i[idx2] * Gy[i+1][j+1];
				//printf("Gy[] = %f", Gy[i+1][j+1]);
			}
		}
		
		Gradient_o[idx] = sqrt(GX*GX+GY*GY);
		//printf("GX = %f GY = %f Gradient = %f", GX, GY, sqrt(GX*GX+GY*GY)); //GX always = 0 and GY always = 499?
		Theta_o[idx] = atan(GX/GY) * 180.0/PI;
	}
	End:;
}

__global__ void Threshold(double *Gradient_i, double *Theta_i, uchar *GPU_o, int M, int N, int ThreshLo, int ThreshHi){ //Threshold values need to be part of the parameters
	int row = blockIdx.x * blockDim.x + threadIdx.x; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	//uchar output = Gradient_i[idx];
	
	uchar PIXVAL;
	double L,H,G,T;
	
	/*
	int row46, col46, idx46; //left right
	int row28, col28, idx28; //top bottom
	int row19, col19, idx19; //lower left upper right
	int row37, col37, idx37; //lower right upper left
	*/
	
	if ((row<1) || (row>(M-2))) goto End;
	//col = 1;
	L = (double) ThreshLo; H = (double)ThreshHi;
	G = Gradient_i[idx];
	PIXVAL = NOEDGE;
	if (G <= L){
		PIXVAL = NOEDGE;
	} else if (G >= H){
		//printf("G = %f and H = %f", G, H);
		PIXVAL = EDGE;
	} else{
			//printf("GOT IN HERE?");
		T = Theta_i [idx];
		if ((T < -67.5) || (T > 67.5)){
			//look left and right
			PIXVAL = ((Gradient_i[row*N + col - 1] > H) || (Gradient_i[row*N + col + 1] > H)) ? EDGE:NOEDGE;
		} else if ((T >= -22.5) && (T <= 22.5)){
			//look top and bottom
			PIXVAL = ((Gradient_i[(row - 1)*N + col] > H) || (Gradient_i[(row + 1)*N + col] > H)) ? EDGE:NOEDGE;
		} else if ((T > 22.5) && (T <= 67.5)){
			//look upper right and lower left
			PIXVAL = ((Gradient_i[(row - 1)*N + col + 1] > H) || (Gradient_i[(row + 1)*N + col - 1] > H)) ? EDGE:NOEDGE;
		} else if ((T >= -67.5) && (T < -22.5)){
			//look upper left and lower right
			PIXVAL = ((Gradient_i[(row - 1)*N + col - 1] > H) || (Gradient_i[(row + 1)*N + col + 1] > H)) ? EDGE:NOEDGE;
		}
	}
	//printf("pixval = %d", PIXVAL);
	GPU_o[idx] = PIXVAL;
	End:;
}

int main(int argc, char *argv[]){
	float GPURuntimes[4]; //run times of the GPU code
	float ExecTotalTime, TfrCPUGPU, GPUTotalTime, TfrGPUCPU;
	cudaError_t cudaStatus;
	char filename[100]; //output file name
	//int i;
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
	//for (i = 0; i < NumIter; i++){

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
		sprintf(filename, "%s.bmp",argv[2]);//,i);
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
	//}
	exit(EXIT_SUCCESS);
		
}

cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes){
	
	cudaEvent_t time1, time2, time3, time4;
	int ucharGPUSize, doubleGPUSize; // total size of 1 image in bytes
	uchar *GPU_idata;
	uchar *GPU_odata;
	uchar *GPU_zerodata;
	double *GPU_Gaussdata;
	double *GPU_Gradientdata;
	double *GPU_Thetadata;
	
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
	doubleGPUSize = M * N * sizeof(double);
	
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
	cudaStatus = cudaMalloc((void**)&GPU_Gaussdata, doubleGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_Gradientdata, doubleGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_Thetadata, doubleGPUSize);
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
	GaussianFilter<<<numBlocks, threadsPerBlock>>>(GPU_idata, GPU_Gaussdata, M, N);

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
	

	
	// THEN SOBEL THEN THRESHOLD
	Sobel<<<numBlocks, threadsPerBlock>>>(GPU_Gaussdata, GPU_Gradientdata, GPU_Thetadata, M, N);

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
	
	Threshold<<<numBlocks, threadsPerBlock>>>(GPU_Gradientdata, GPU_Thetadata, GPU_odata, M, N, 8, 15);
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
	cudaFree(GPU_Gaussdata);
	cudaFree(GPU_Gradientdata);
	cudaFree(GPU_Thetadata);
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
	//ThreshLo++;
	return cudaStatus;
}
