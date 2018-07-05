#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
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

#define BOX_SIZE2			16			// ThreadsPerBlock == BOX_SIZE * BOX_SIZE
#define BOX_SIZE1			1

int M; //number of rows in image
int N; //number of columns in image
int NumRot;
int Mode;
int a = 0;
Mat zero;

//ip.Vpixels <--> M
//ip.Hpixels <--> N

__global__ void rotate_kernel1(uchar *GPU_i, uchar *GPU_o, int M, int N, int i, int j){

	//Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	//Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	__shared__ uchar shared_GPU_data[BOX_SIZE1][BOX_SIZE1];

	int row = bx * BOX_SIZE1 + tx; //row of image
	int col = by * BOX_SIZE1 + ty; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	shared_GPU_data[tx][ty] = GPU_i[idx];
	
	__syncthreads();

    int h,v,c;
	int row2; //new row of image
	int col2; //new column of image

	double X, Y, newY, newX, ScaleFactor;
	double Diagonal, H, V;
	double RotDegrees = 360 / j * i; //in degrees
	double RotAngle = 2*3.141592/360.000*(double) RotDegrees; //in radians
	//printf("We are rotating %d times and iteration# = %d RotAngle = %g\n", j, i, RotAngle);
	// transpose image coordinates to Cartesian coordinates
	// integer div
	c = col;
	h=N/2; 	//halfway of column pixels
	v=M/2;	//halfway of horizontal pixels
	X=(double)c-(double)h;
	Y=(double)v-(double)row;
	
	// pixel rotation matrix	
	newX = cos(RotAngle) * X - sin(RotAngle) * Y;
	newY= sin (RotAngle) * X + cos(RotAngle) * Y;

	
	// Scale to fit everything in the image box CONFIRMED TO BE CORRECT
	H=(double)N;
	V=(double)M;
	Diagonal=sqrt(H*H+V*V);
	ScaleFactor=(N>M) ? V/Diagonal : H/Diagonal;
	
	newX=newX*ScaleFactor;
	newY = newY*ScaleFactor;
	// convert back from Cartesian to image coordinates

	col2= (int)newX+h;
	row2=v-(int)newY;
	// maps old pixel to new pixel
	int idx2 = row2*N + col2;
	GPU_o[idx2] = shared_GPU_data[tx][ty];

}


__global__ void rotate_kernel2(uchar *GPU_i, uchar *GPU_o, int M, int N, int i, int j){

	//Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	//Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	__shared__ uchar shared_GPU_data[BOX_SIZE2][BOX_SIZE2];

	int row = bx * BOX_SIZE2 + tx; //row of image
	int col = by * BOX_SIZE2 + ty; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	shared_GPU_data[tx][ty] = GPU_i[idx];
	
	__syncthreads();

    int h,v,c;
	int row2; //new row of image
	int col2; //new column of image

	double X, Y, newY, newX, ScaleFactor;
	double Diagonal, H, V;
	double RotDegrees = 360 / j * i; //in degrees
	double RotAngle = 2*3.141592/360.000*(double) RotDegrees; //in radians
	//printf("We are rotating %d times and iteration# = %d RotAngle = %g\n", j, i, RotAngle);
	// transpose image coordinates to Cartesian coordinates
	// integer div
	c = col;
	h=N/2; 	//halfway of column pixels
	v=M/2;	//halfway of horizontal pixels
	X=(double)c-(double)h;
	Y=(double)v-(double)row;
	
	// pixel rotation matrix	
	newX = cos(RotAngle) * X - sin(RotAngle) * Y;
	newY= sin (RotAngle) * X + cos(RotAngle) * Y;

	
	// Scale to fit everything in the image box CONFIRMED TO BE CORRECT
	H=(double)N;
	V=(double)M;
	Diagonal=sqrt(H*H+V*V);
	ScaleFactor=(N>M) ? V/Diagonal : H/Diagonal;
	
	newX=newX*ScaleFactor;
	newY = newY*ScaleFactor;
	// convert back from Cartesian to image coordinates

	col2= (int)newX+h;
	row2=v-(int)newY;
	// maps old pixel to new pixel
	int idx2 = row2*N + col2;
	GPU_o[idx2] = shared_GPU_data[tx][ty];

}

__global__ void rotate_kernel3(uchar *GPU_i, uchar *GPU_o, int M, int N, int i, int j){
//    int row,col,h,v,c;
    int row,col,h,v,hp3;
	double cc, ss, k1, k2;
	int NewRow,NewCol;
	double Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
	double CRA,SRA, CRAS, SRAS, SRAYS, CRAYS;

	double RotDegrees = 360 / j * i; //in degrees
	double RotAngle = 2*3.141592/360.000*(double) RotDegrees; //in radians
	//printf("We are rotating %d times and iteration# = %d RotAngle = %g\n", j, i, RotAngle);
	// transpose image coordinates to Cartesian coordinates

	//Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	//Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	__shared__ uchar shared_GPU_data[BOX_SIZE1][BOX_SIZE1];

	int row = bx * BOX_SIZE1 + tx; //row of image
	int col = by * BOX_SIZE1 + ty; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	shared_GPU_data[tx][ty] = GPU_i[idx];
	
	__syncthreads();

	
			H=(double) N;
			V=(double) M;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(N>M) ? V/Diagonal : H/Diagonal;
			
			CRA=cos(RotAngle);	CRAS=ScaleFactor*CRA;
			SRA=sin(RotAngle);	SRAS=ScaleFactor*SRA;
// integer div
	h=N/2; 	//halfway of column pixels
	v=M/2;	//halfway of horizontal pixels
        col=0;
		cc=0.00;
		ss=0.00;
			Y=(double)v-(double)row;
			SRAYS=SRAS*Y;     CRAYS=CRAS*Y;
			k1=CRAS*(double)h + SRAYS;
			k2=SRAS*(double)h - CRAYS;

			// transpose image coordinates to Cartesian coordinates
			
			// pixel rotation matrix
			newX=cc-k1;
			newY=ss-k2;

	
			
			// convert back from Cartesian to image coordinates
			col2=((int) newX+h);
			row2=v-(int)newY;     
	/*		if((NewCol>=0) && (NewRow>=0) && (NewCol<ip.Hpixels) && (NewRow<ip.Vpixels)){

            }*/
	// maps old pixel to new pixel
	int idx2 = row2*N + col2;
	GPU_o[idx2] = shared_GPU_data[tx][ty];
}

int main(int argc, char *argv[]){

	float GPURuntimes[4]; 	// run times of the GPU code
	float ExecTotalTime, GPUTotalTime;
	cudaError_t cudaStatus;
	char filename[100]; //output file name
	int i;

	int *CPU_OutputArray = (int*) 0; 	// where the GPU should copy the output back to
	
	if (argc != 5){
		printf("Improper usage!\n");
		printf("Usage: %s <input image> <output image> <N rotations> <mode [1-5]>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	NumRot = atoi(argv[3]);
	if (NumRot > 30){
		printf("Number of rotations requested is too high!  Adjusted to 30.\n");
		NumRot = 30;
	}
	Mode = atoi(argv[4]);
	if (Mode >=6){
		printf("Improper usage! Mode must be between [1-5]\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i<NumRot; i++){	
	// Load image:
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
	// Create CPU memory to store the output;
	/*Mat */zero = Mat(M,N,CV_8UC1, Scalar(255)); //start by making every pixel white
	sprintf(filename,"%sAROT%d.png", argv[2], i);
	imwrite(filename,zero);
	CPU_OutputArray = (int*) malloc(M*N*sizeof(int));
	if (CPU_OutputArray == NULL){
		fprintf(stderr, "OOPS.  Can't create CPU_OutputArray using malloc() ...\n");
		exit(EXIT_FAILURE);
	}
	//run it

	cudaStatus = launch_helper(image, CPU_OutputArray, GPURuntimes);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "launch_helper failed!\n");
		free(CPU_OutputArray);
		exit(EXIT_FAILURE);
	}
	
	printf("-----------------------------------------------------------------\n");
	printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \nSum of Iteration = %5.2f ms\n",
			GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
	ExecTotalTime += GPURuntimes[0];
	GPUTotalTime += GPURuntimes[2];
	printf("\nGPU Execution Time = %5.2f ms \n", GPUTotalTime);
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
	int TotalGPUSize; // total size of 1 image in bytes
	uchar *GPU_idata;
	uchar *GPU_odata;
	uchar *GPU_zerodata;
	
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
	TotalGPUSize = M * N * sizeof(uchar);
	cudaStatus = cudaMalloc((void**)&GPU_idata, TotalGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_odata, TotalGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_zerodata, TotalGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPU_odata, zero.data, TotalGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyzero failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPU_idata, image.data, TotalGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaEventRecord(time2, 0);

	
/*	for (int row=0;row<image.rows;row++) 
{ 
   for (int col=0;col<image.cols;col++) 
    {
       image.at<uchar>(row,col) = 0;
    } 
}*/
	// Launch a kernel on the GPU with one thread for each pixel.
	if (Mode ==1){
	threadsPerBlock = dim3(BOX_SIZE1, BOX_SIZE1);
	numBlocks = dim3(M / threadsPerBlock.x, N / threadsPerBlock.y);
		rotate_kernel1<<<numBlocks, threadsPerBlock>>>(GPU_idata, GPU_odata, M, N, a, NumRot);
	}
	else if (Mode == 2){
		threadsPerBlock = dim3(BOX_SIZE2, BOX_SIZE2);
		numBlocks = dim3(M / threadsPerBlock.x, N / threadsPerBlock.y);
		rotate_kernel2<<<numBlocks, threadsPerBlock>>>(GPU_idata, GPU_odata, M, N, a, NumRot);
	}
	else if (Mode == 3){
		threadsPerBlock = dim3(BOX_SIZE1, BOX_SIZE1);
		numBlocks = dim3(M / threadsPerBlock.x, N / threadsPerBlock.y);
		rotate_kernel3<<<numBlocks, threadsPerBlock>>>(GPU_idata, GPU_odata, M, N, a, NumRot);
	}
		// Check for errors immediately after kernel launch.
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
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
		cudaStatus = cudaMemcpy(CPU_OutputArray, GPU_odata, TotalGPUSize, cudaMemcpyDeviceToHost);
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
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);

	a++;
	return cudaStatus;
}
