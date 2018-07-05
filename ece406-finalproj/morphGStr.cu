/**
VERSION 3: 	SMALL CHANGES BASED ON VERSION 2 PROVIDED BY FRANCO 12:58 6 DEC
			ATTEMPTED TO FIX LOGICAL ISSUES WITH LEVELS SEE LINE 430(END OF BRACKET AT 443) AND LINE 444(END OF BRACKET AT 487)
			THE FOLLOWING ISSUE WAS FOUND: 
			> CUDA Runtime Error: an illegal memory access was encountered
			
			UPON ONLY RUNNING DILATION AND COMMENTING OUT EROSION AND DIFFERENCE, THE FOLLOWING WAS FOUND:
			> OpenCV Error: Unspecified error (could not find a writer for the specified extension) in imwrite_, file /software/opencv/src/opencv-2.4.11/modules/highgui/src/loadsave.cpp, line 275
			> terminate called after throwing an instance of 'cv::Exception'
			> what():  /software/opencv/src/opencv-2.4.11/modules/highgui/src/loadsave.cpp:275: error: (-2) could not find a writer for the specified extension in function imwrite_



**/

#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//CUDA STUFF:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

//OpenCV stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

cudaError_t launch_helper(uchar *CPU_InputArray, uchar *CPU_OutputArray, float* Runtimes);

inline
cudaError_t checkCuda(cudaError_t result,int line)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s at line : %d\n", cudaGetErrorString(result),line);
    // We should be free()ing CPU+GPU memory here, but we're relying on the OS
    // to do it for us.
    cudaDeviceReset();
    assert(result == cudaSuccess);
  }
  return result;
}
/*
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    // We should be free()ing CPU+GPU memory here, but we're relying on the OS
    // to do it for us.
    cudaDeviceReset();
    assert(result == cudaSuccess);
  }
  return result;
}
*/
#define BOX_SIZE		16 		//ThreadsPerBlock == BOX_SIZE * BOX_SIZE
#define PI				3.1415926
#define EDGE			255
#define NOEDGE			0

int M; //number of rows in the image
int N; //number of columns in the image
uchar *CPU_InputArray;
uchar *CPU_OutputArray; //where the GPU should copy the output back to

char Type;
int Xo; // Width of structuring element
int Yo; // Height of structuring element
uchar * StrElement;
int TotalSize;
int NumIter = 16;
int Thresh = 32;
int nStreams = 4;
int levels = 16;

Mat zero;

//ip.Vpixels <--> M
//ip.Hpixels <--> N

/*__device__ uchar StrElement[3][3] = { 	{	0, 	1, 	0	},
										{	1,	1,	1	},
										{	0,	1,	0	}	};
*/

//kernels

void CreateStrElement() {
	int i, j;
	if (Type=='S') {
		for (i = 0; i<Yo;i++) {
			for (j = 0; j < Xo; j++) {
				StrElement[i*Xo+j] = 1;
			}
		}
	}
	if (Type=='C') {
		int R = sqrt(Xo*Xo/4+Yo*Yo/4);
		int H;
		for (i = -Yo/2; i<Yo/2; i++) {
			H = sqrt(R*R-Yo*Yo/4);
			for (j = -H; j<H; j++) {
				StrElement[(i+Yo/2)*Xo+(j+Xo/2)] = 1;
			}
		}
	}
	if (Type=='X') {
		for (i = 0; i<Yo;i++) {
			if (i==Yo/2) {
				for (j = 0; j < Xo; j++) {
					StrElement[i*Xo+j] = 1;
				}
			}
		}
	}

}

__global__ void Erosion(uchar *GPU_i, uchar *Erosiondata, uchar *StrElement, int M, int N, int Xo, int Yo, int offsetx, int offsety){

	int row = blockIdx.x * blockDim.x + threadIdx.x+offsetx; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y+offsety; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	//uchar output = GPU_i[idx];
	//uchar ElementResult;
	int min;
	int i,j;
	int d = Yo/2;
	int e = Xo/2;
	min = 255;
	if ((row < d) || (row > (M - d-1))) goto End;
	int idx2, row2, col2;
	for (i=-d; i<=d; i++){
		for (j=-e; j<=e; j++){
			if (min == 0) continue;
			row2 = row + i;
			col2 = col + j;
			idx2 = row2*N + col2;
			/*if (StrElement[(i+Yo/2)*Xo+j+Xo/2] == 0){
				ElementResult = 255;
				if  (min > ElementResult)
					min = ElementResult;
				//printf("got here 0");
			}*/
			if (StrElement[(i+Yo/2)*Xo+j+Xo/2] == 1){
				//ElementResult = GPU_i[idx2];
				if  (min > GPU_i[idx2])
					min = GPU_i[idx2];
			}
		}
	}

	Erosiondata [idx] = min;
	
	End:;
}

__global__ void Dilation(uchar *GPU_i, uchar *Dilationdata, uchar *StrElement, int M, int N, int Xo, int Yo, int offsetx, int offsety){
	int row = blockIdx.x * blockDim.x + threadIdx.x+offsetx; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y+offsety; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	//uchar output = GPU_i[idx];
	//uchar ElementResult;
	int max;
	int i,j;
	int d = Yo/2;
	int e = Xo/2;
	max = 0;
	if ((row < d) || (row > (M - d-1))) goto End;
	int idx2, row2, col2;
	for (i=-d; i<=d; i++){
		for (j=-e; j<=e; j++){
			if (max == 255) continue;
			row2 = row + i;
			col2 = col + j;
			idx2 = row2*N + col2;
			/*if (StrElement[(i+Yo/2)*Xo+j+Xo/2] == 0){
				ElementResult = 0;
				if  (max < ElementResult)
					max = ElementResult;
				//printf("got here 0");
			}*/
			if (StrElement[(i+Yo/2)*Xo+j+Xo/2] == 1){
				//ElementResult = GPU_i[idx2];
				if  (max < GPU_i[idx2])
					max = GPU_i[idx2];
			}
		}
	}

	Dilationdata [idx] = max;

	End:;

}


__global__ void Threshold(uchar *GPU_i, uchar *GPU_o, int M, int N, int Thresh, int offsetx, int offsety)
{
    //long tn;            		     // My thread number (ID) is stored here
    //int row,col;
	unsigned char PIXVAL;
	double L,G;

    int rt = blockIdx.x * blockDim.x + threadIdx.x+offsetx;  // row of image
	int ct = blockIdx.y * blockDim.y + threadIdx.y+offsety;  // column of image
	//int k;
	int idx = rt*N+ct;  // which pixel in full 1D array
	if (rt>M-1 || ct>N-1) {
		//GPU_o[idx] = NOEDGE;
		return;
	}

	L=(double)Thresh;		//H=(double)ThreshHi;
	G=GPU_i[idx];
	PIXVAL=NOEDGE;
	if(G<=L){						// no edge
		PIXVAL=NOEDGE;
	}
	else {					// edge
		PIXVAL=EDGE;
	}

	GPU_o[idx]=PIXVAL;

}


__global__ void Difference(uchar *Dilationdata, uchar *Erosiondata, uchar *GPU_o, int M, int N, int offsetx, int offsety){
	int row = blockIdx.x * blockDim.x + threadIdx.x+offsetx; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y+offsety; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	int D = Dilationdata[idx];
	int E = Erosiondata[idx];
	GPU_o [idx] = D - E;

}

void show_image(Mat image, string title) {
  //if (1) {
    namedWindow(title, WINDOW_AUTOSIZE);  // create window
    imshow(title, image);                 // show image
    waitKey(0);
  //}
}
	
int main(int argc, char *argv[]){
	float GPURuntimes[4]; //run times of the GPU code
	float ExecTotalTime, TfrCPUGPU, GPUTotalTime, TfrGPUCPU;
	cudaError_t cudaStatus;
	//; //output file name
	int i = 1;

	
	if (argc != 6){
		printf("Improper Usage!\n");
		printf("Usage: %s <input image> <output image> <S,X,C> <Width of StrEl> <Height of StrEl>\n", argv[0]);
		printf("Where: S is square-shaped StrEl, X is cross-shaped StrEl, and C is circular StrEl.\n");
		exit(EXIT_FAILURE);
	}
	Type = argv[3][0];
	Xo = atoi(argv[4]);
	Yo = atoi(argv[5]);

	if (Type == 'C' && Xo != Yo) {
		printf("Error: For circles, StrEl width must equal StrEl height.\n");
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
	TotalSize = M * N * sizeof(uchar);
	//Create CPU memory to store the output;
	//zero = Mat(M,N,CV_8UC1, Scalar(255)); //start by making every pixel white
	//sprintf(filename, "%s%d.png",argv[2],i);
	//imwrite(filename, zero);
	checkCuda(cudaMallocHost((void**)&StrElement, Xo*Yo*sizeof(uchar)),__LINE__);
	checkCuda(cudaMallocHost((void**)&CPU_InputArray, TotalSize),__LINE__);
	memcpy(CPU_InputArray, image.data, TotalSize);  // always the same image
	//  Allocate the output while we're at it:
	checkCuda(cudaMallocHost((void**)&CPU_OutputArray, TotalSize),__LINE__);

	CreateStrElement();

	//run it
	checkCuda(launch_helper(CPU_InputArray, CPU_OutputArray, GPURuntimes),__LINE__);
	/*if (cudaStatus != cudaSuccess){
		fprintf(stderr, "launch_helper failed!\n");
		free(CPU_OutputArray);
		exit(EXIT_FAILURE);
	}*/
	// FIX THIS LAST
	printf("-----------------------------------------------------------------\n");
	printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \nSum of Iteration = %5.2f ms\n",
			GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
	/*ExecTotalTime += GPURuntimes[0];
	TfrCPUGPU += GPURuntimes[1];
	GPUTotalTime += GPURuntimes[2];
	TfrGPUCPU += GPURuntimes[3];
	printf("\nTotal Tfr CPU -> GPU Time = %5.2f ms\n", TfrCPUGPU);
	printf("GPU Execution Time = %5.2f ms \n", GPUTotalTime);
	printf("Total Tfr GPU -> CPU Time = %5.2f ms\n", TfrGPUCPU);
	printf("Total Execution Time = %5.2f ms\n", ExecTotalTime);*/
	printf("-----------------------------------------------------------------\n");

	/*cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaDeviceReset failed!\n");
		cudaFreeHost(StrElement);
		cudaFreeHost(CPU_InputArray);
		cudaFreeHost(CPU_OutputArray);
		exit(EXIT_FAILURE);
	}*/

	Mat result = Mat(M, N, CV_8UC1, CPU_OutputArray);

	//save image to disk
	string output_filename = argv[2];
	if (!imwrite(output_filename, image)) {
		fprintf(stderr, "couldn't write output to disk!\n");
		//free();
		exit(EXIT_FAILURE);
	}

	//printf("i : %d\n",i);
	char n0, n1;
	if (i>9) {
		n0 = '1';
		n1 = (i-10)+'0';
		output_filename.insert(output_filename.end()-4,n0);
		output_filename.insert(output_filename.end()-4,n1);
	}

	else {
		n0 = i+'0';
		output_filename.insert(output_filename.end()-4,n0);
	}

	//show_image(result, output_filename);

	//printf("output: %s\n", output_filename.c_str());
	//output_filename[strl-5] = n;
	if (!imwrite(output_filename, result)) {
		fprintf(stderr, "couldn't write output to disk!\n");
		cudaFreeHost(StrElement);
		cudaFreeHost(CPU_InputArray);
		cudaFreeHost(CPU_OutputArray);
		exit(EXIT_FAILURE);
	}
	printf("Saved image '%s', size = %dx%d (dims = %d).\n",
		   output_filename.c_str(), result.rows, result.cols, result.dims);
	cudaFreeHost(StrElement);
	cudaFreeHost(CPU_InputArray);
	cudaFreeHost(CPU_OutputArray);
	checkCuda( cudaDeviceReset(), __LINE__ );
	//}
	exit(EXIT_SUCCESS);
		
}

cudaError_t launch_helper(uchar *CPU_InputArray, uchar *CPU_OutputArray, float* Runtimes){
	
	cudaEvent_t time1, time2, time3, time4;
	int ucharGPUSize; // total size of 1 image in bytes
	//int offsetx, offsety;
	int TotalSize_2 = (M / levels + Xo/2)*N*sizeof(uchar);
	uchar *GPU_idata;
	uchar *GPU_odata;
	//uchar *GPU_zerodata;
	uchar *GPU_Dilationdata;
	uchar *GPU_Erosiondata;
	
	uchar *GPU_StrElement;

	dim3 threadsPerBlock;
	dim3 numBlocks;
	dim3 sharedBlocks;
	dim3 streamSize;
	
	
	cudaError_t cudaStatus;
	checkCuda(cudaSetDevice(0), __LINE__); // use the first GPU

	threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
	numBlocks = dim3(ceil((float)M / threadsPerBlock.x),ceil((float)N / threadsPerBlock.y));
	sharedBlocks = dim3(ceil((float)numBlocks.x / levels), ceil((float)numBlocks.y / nStreams));

	cudaStream_t stream[nStreams];
	//checkCuda( cudaEventCreate(&startEvent) );
	//checkCuda( cudaEventCreate(&stopEvent) );
	for (int i = 0; i < nStreams; ++i) {
		checkCuda(cudaStreamCreate(&stream[i]),__LINE__);
	}
	
	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

	cudaEventRecord(time1, 0);
	
	// Allocate GPU buffer for inputs and outputs:
	ucharGPUSize = M * N * sizeof(uchar);
	
	checkCuda( cudaMalloc((void**)&GPU_idata, ucharGPUSize), __LINE__);

	checkCuda(cudaMalloc((void**)&GPU_odata, ucharGPUSize), __LINE__);
	
	checkCuda(cudaMalloc((void**)&GPU_Dilationdata, ucharGPUSize), __LINE__);

	checkCuda( cudaMalloc((void**)&GPU_Erosiondata, ucharGPUSize), __LINE__);


	checkCuda( cudaMalloc((void**)&GPU_StrElement, Xo*Yo*sizeof(uchar)),__LINE__);

	checkCuda(cudaMemset(GPU_Dilationdata, 0, ucharGPUSize),__LINE__);

	checkCuda(cudaMemset(GPU_Erosiondata, 0, ucharGPUSize),__LINE__);

	checkCuda(cudaMemset(GPU_odata, 0, ucharGPUSize),__LINE__);

	checkCuda(cudaMemcpy(GPU_StrElement, StrElement, Xo*Yo*sizeof(uchar), cudaMemcpyHostToDevice), __LINE__);
	int offsetx, offsety;
	
	for (int i = 0; i < levels; i++) {
		//if (i < levels + 1) {
			if (i < levels - 1) {
				checkCuda(cudaMemcpyAsync(&GPU_idata[ucharGPUSize / levels*i], &CPU_InputArray[ucharGPUSize/levels* i], TotalSize_2, cudaMemcpyHostToDevice, stream[0]), __LINE__);
			}
			if (i == levels-1) {
				checkCuda(cudaMemcpyAsync(&GPU_idata[ucharGPUSize / levels*i], &CPU_InputArray[ucharGPUSize/levels*i], ucharGPUSize/levels, cudaMemcpyHostToDevice, stream[0]), __LINE__);
			}
			cudaEventRecord(time2, 0);
			//printf("Copying levels: %d to %d\n",ucharGPUSize/levels*i, ucharGPUSize/levels*(i+1));
			// Launch a kernel on the GPU with one thread for each pixel.


			//if (i > 0){

			//EROSION AND DILATION
			offsetx = threadsPerBlock.x*sharedBlocks.x*(i);

			for (int j = 0; j < nStreams; j++) {
				offsety = j*sharedBlocks.y*threadsPerBlock.y;
				Dilation << <sharedBlocks, threadsPerBlock, 0, stream[j] >> >(GPU_idata, GPU_Dilationdata, GPU_StrElement, M, N, Xo, Yo, offsetx, offsety);

				// Check for errors immediately after kernel launch.
				checkCuda(cudaGetLastError(), __LINE__);
			}
				// cudaDeviceSynchronize waits for the kernel to finish, and returns
				// any errors encountered during the launch.
			for (int j = 0; j < nStreams; j++){
				offsety = j*sharedBlocks.y*threadsPerBlock.y;
				Erosion << <sharedBlocks, threadsPerBlock, 0, stream[j] >> >(GPU_idata, GPU_Erosiondata, GPU_StrElement, M, N, Xo, Yo, offsetx, offsety);

				// Check for errors immediately after kernel launch.
				checkCuda(cudaGetLastError(), __LINE__);

				// cudaDeviceSynchronize waits for the kernel to finish, and returns
				// any errors encountered during the launch.

			}
		//}
		//if (i > 0) {
			// THEN TAKE THE DIFFERENCE
			offsetx = threadsPerBlock.x*sharedBlocks.x*(i);
			for (int j = 0; j < nStreams; j++){
				offsety = j*sharedBlocks.y*threadsPerBlock.y;
				Difference << <sharedBlocks, threadsPerBlock, 0, stream[j] >> >(GPU_Dilationdata, GPU_Erosiondata, GPU_odata, M, N, offsetx, offsety);

				// Check for errors immediately after kernel launch.
				checkCuda(cudaGetLastError(), __LINE__);
			}

			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.



			cudaEventRecord(time3, 0);
			// Copy output (results) from GPU buffer to host (CPU) memory.
			checkCuda(cudaMemcpyAsync(&CPU_OutputArray[ucharGPUSize / levels*(i)],
					&GPU_odata[ucharGPUSize / levels*(i)], ucharGPUSize / levels,
					cudaMemcpyDeviceToHost, stream[0]), __LINE__);
			cudaEventRecord(time4, 0);
		//}
	}

	checkCuda(cudaDeviceSynchronize(),__LINE__);
	
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

	for (int i = 0; i < nStreams; ++i) {
		checkCuda(cudaStreamDestroy(stream[i]),__LINE__);
	}
	cudaFree(GPU_odata);
	cudaFree(GPU_idata);
	cudaFree(GPU_Dilationdata);
	cudaFree(GPU_Erosiondata);

	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
	//ThreshLo++;
	return cudaSuccess;
}
