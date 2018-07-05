#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

//CUDA STUFF:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//OpenCV stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#define BOX_SIZE		16 		//ThreadsPerBlock == BOX_SIZE * BOX_SIZE
#define PI				3.1415926
#define EDGE			255
#define NOEDGE			0
#define ONEOVER159   0.00628931
#define MAXTHREADS 		64

int M; //number of rows in the image
int N; //number of columns in the image
int TotalSize;
int NumIter = 16;
int ThreshLo = 0;
Mat zero;
int stream = 4; //allocating the work to be done among CPU and GPU
int levels = 8; //granularity

// only relevant for the portion of image CPU is assigned to
long  			NumThreads = 2;  			// Regular/Pre-processing threads
int 	     	ThParam[MAXTHREADS];		// Thread parameters ...
pthread_t      	ThHandle[MAXTHREADS];		// Thread handles
pthread_attr_t 	ThAttr;						// Pthread attrributes

uchar *CPU_InputArray;
uchar *CPU_OutputArray;
double *CPU_Gauss;				// Gauss filtered version of the B&W image
double *CPU_Gradient, *CPU_Theta;		// gradient and theta for each pixel
uchar *CPU_Threshold;
int TotalGPUSize;

cudaError_t launch_helper(uchar *CPU_InputArray, uchar *CPU_OutputArray, float *GPURuntimes);

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


//ip.Vpixels <--> M
//ip.Hpixels <--> N

//kernels
__global__ void GaussianFilter(uchar *GPU_i, double *Gauss_o, int M, int N, int offsetx, int offsety){

	int row = blockIdx.x * blockDim.x + threadIdx.x + offsetx; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y + offsety; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	int i,j;
	double G;
	double Gauss[5][5] = {	{ 2, 4,  5,  4,  2 },
						{ 4, 9,  12, 9,  4 },
						{ 5, 12, 15, 12, 5 },
						{ 4, 9,  12, 9,  4 },
						{ 2, 4,  5,  4,  2 }	};
	
	if ((row < 2) || (row > (M - 3))) goto End;
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

void *PThrGaussianFilter(void* tid){

	long tn;
	int row,col,i,j;
	float G;
	
		double Gauss[5][5] = {	{ 2, 4,  5,  4,  2 },
						{ 4, 9,  12, 9,  4 },
						{ 5, 12, 15, 12, 5 },
						{ 4, 9,  12, 9,  4 },
						{ 2, 4,  5,  4,  2 }	};
	
	tn = *((int *) tid);
	tn *= M/NumThreads/levels + (levels - 1)*M/levels; //assign CPU to the last one
	
    for(row=tn; row<tn+M/NumThreads/levels; row++){
		if((row<2) || (row>(M-3))) continue;
        col=2;
        while(col<=(N-3)){
			G=0.0;
			for(i=-2; i<=2; i++){
				for(j=-2; j<=2; j++){
					G = G + CPU_InputArray[(row+i)*N + col+j]*Gauss[i+2][j+2];
				}
			}
			CPU_Gauss[row*N + col]=G*ONEOVER159;
			//printf("CPU_Gauss[] = %f", CPU_Gauss[row*N + col]);
            col++;
        }
    }
	
	pthread_exit(NULL);
}

__global__ void Sobel(double *Gauss_i, double *Gradient_o, double *Theta_o, int M, int N, int offsetx, int offsety){
	int row = blockIdx.x * blockDim.x + threadIdx.x + offsetx; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y + offsety; //column of image
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
		//printf("GX = %f GY = %f Gradient = %f", GX, GY, sqrt(GX*GX+GY*GY));
		Theta_o[idx] = atan(GX/GY) * 180.0/PI;
	}
	End:;
}

void *PThrSobel(void* tid){
double Gx[3][3] = {		{ -1, 0, 1 },
						{ -2, 0, 2 },
						{ -1, 0, 1 }	};

double Gy[3][3] = {		{ -1, -2, -1 },
						{  0,  0,  0 },
						{  1,  2,  1 }	};
    long tn;            		     // My thread number (ID) is stored here
    int row,col,i,j;
	float GX,GY;

	tn = *((int *) tid);           // Calculate my Thread ID
	tn *= M/NumThreads/levels + (levels - 1)*M/levels; //assign CPU to the last one

	for(row=tn; row<tn+M/levels/NumThreads; row++){
		if((row<1) || (row>(M-2))) continue;
        col=1;
        while(col<=(N-2)){
			// calculate Gx and Gy
			GX=0.0; GY=0.0;
			for(i=-1; i<=1; i++){
				for(j=-1; j<=1; j++){
					GX+=CPU_Gauss[(row+i)*N+col+j]*Gx[i+1][j+1];
					GY+=CPU_Gauss[(row+i)*N+col+j]*Gy[i+1][j+1];
				}
			}
			CPU_Gradient[row*N+col]=sqrt(GX*GX+GY*GY);
			//printf("CPU_Gradient[] = %f", CPU_Gradient[row*N + col]);
			CPU_Theta[row*N+col] = atan(GX/GY) * 180.0/PI;
            col++;
        }
    }
    pthread_exit(NULL);
}

__global__ void Threshold(double *Gradient_i, double *Theta_i, uchar *GPU_o, int M, int N, int offsetx, int offsety, int ThreshLo, int ThreshHi){ //Threshold values need to be part of the parameters
	int row = blockIdx.x * blockDim.x + threadIdx.x + offsetx; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y + offsety; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	//uchar output = Gradient_i[idx];
	
	uchar PIXVAL;
	double L,H,G,T;
	
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

void *PThrThreshold(void* tid){
    long tn;            		     // My thread number (ID) is stored here
    int row,col;
	unsigned char PIXVAL;
	double L,G,T,H;

	tn = *((int *) tid);           // Calculate my Thread ID
	tn *= M/NumThreads/levels + (levels - 1)*M/levels; //assign CPU to the last one
	//printf("row = %d -> %d\n\n\n", tn , tn+M/levels/NumThreads);
	
	for(row=tn; row<tn+M/levels/NumThreads; row++){
		if((row<1) || (row>(M-2))) continue;
        col=1;
        while(col<=(N-2)){
			L = (double) ThreshLo;		H = L + 8;
			G = CPU_Gradient[row*N+col];
			//printf("CPU_Gradient[] = %f", CPU_Gradient[row*N + col]);
			PIXVAL = NOEDGE;
			if(G<=L){						// no edge
				PIXVAL=NOEDGE;
			}else if(G>=H){					// edge
				PIXVAL=EDGE;
			}else{
				T=CPU_Theta[row*N + col];
				//printf("T[] = %f", T);
				if((T<-67.5) || (T>67.5)){   
					// Look at left and right
					PIXVAL=((CPU_Gradient[row*N + col-1]>H) || (CPU_Gradient[row * N + col+1]>H)) ? EDGE:NOEDGE;
				}else if((T>=-22.5) && (T<=22.5)){  
					// Look at top and bottom
					PIXVAL=((CPU_Gradient[(row-1) * N + col]>H) || (CPU_Gradient[(row+1) * N + col]>H)) ? EDGE:NOEDGE;
				}else if((T>22.5) && (T<=67.5)){   
					// Look at upper right, lower left
					PIXVAL=((CPU_Gradient[(row-1) * N + col+1]>H) || (CPU_Gradient[(row+1) * N +col-1]>H)) ? EDGE:NOEDGE;
				}else if((T>=-67.5) && (T<-22.5)){   
					// Look at upper left, lower right
					PIXVAL=((CPU_Gradient[(row-1) * N  + col-1]>H) || (CPU_Gradient[(row+1) * N + col+1]>H)) ? EDGE:NOEDGE;
				}
			}
			CPU_Threshold[row*N + col] = 255 - PIXVAL;
			//printf("CPU_Threshold[] = %f", CPU_Threshold[row*N + col]);
            col++;
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char *argv[]){
	float GPURuntimes[4]; //run times of the GPU code
	float ExecTotalTime, TfrCPUGPU, GPUTotalTime, TfrGPUCPU;
	char filename[100]; //output file name
	int i,j, ThErr;

	if (argc != 3){
		printf("Improper Usage!\n");
		printf("Usage: %s <input image> <output image>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	if((NumThreads<0) || (NumThreads>MAXTHREADS)){ //NumThreads handling
	  printf("\nNumber of threads must be between 0 and %u... \n",MAXTHREADS);
	  printf("\n'1' means Pthreads version with a single thread\n");
	  printf("\nYou can also specify '0' which means the 'serial' (non-Pthreads) version... \n\n");
	  printf("\n\nNothing executed ... Exiting ...\n\n");
	  exit(EXIT_FAILURE);
  }
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
	TotalGPUSize = M * N * sizeof(uchar);
	int DoubleTotalGPUSize = M * N * sizeof(double);

	checkCuda(cudaMallocHost( (void**)&CPU_InputArray,  TotalGPUSize), __LINE__); //checkcuda
	memcpy(CPU_InputArray, image.data, TotalGPUSize);  // always the same image
	//  Allocate the output while we're at it:
	checkCuda(cudaMallocHost( (void**)&CPU_OutputArray, TotalGPUSize), __LINE__); //checkcuda
	checkCuda(cudaMallocHost((void**)&CPU_Gauss, DoubleTotalGPUSize), __LINE__);
	checkCuda(cudaMallocHost((void**)&CPU_Gradient, DoubleTotalGPUSize), __LINE__);
	checkCuda(cudaMallocHost((void**)&CPU_Theta, DoubleTotalGPUSize), __LINE__);
	checkCuda(cudaMallocHost((void**)&CPU_Threshold, TotalGPUSize), __LINE__);
	
	
	for (j = 0; j < NumIter; j++){
		//Create CPU memory to store the output;
		zero = Mat(M,N,CV_8UC1, Scalar(255)); //start by making every pixel white
		sprintf(filename, "%s%d.png",argv[2],j);
		imwrite(filename, zero);
	
		//CPU Pthread stuff
		pthread_attr_init(&ThAttr);
		pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);
		

		for(i=0; i<NumThreads; i++){

			ThParam[i] = i;

			ThErr = pthread_create(&ThHandle[i], &ThAttr, PThrGaussianFilter, (void *)&ThParam[i]);
			if(ThErr != 0){
				printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
				exit(EXIT_FAILURE);
			}
			}
		
	for(i=0; i<NumThreads; i++){
		pthread_join(ThHandle[i], NULL);
	}
			for(i=0; i<NumThreads; i++){

			ThParam[i] = i;

			ThErr = pthread_create(&ThHandle[i], &ThAttr, PThrSobel, (void *)&ThParam[i]);
			if(ThErr != 0){
				printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
				exit(EXIT_FAILURE);
			}
			}
	for(i=0; i<NumThreads; i++){
		pthread_join(ThHandle[i], NULL);
	}
			for(i=0; i<NumThreads; i++){

			ThParam[i] = i;

			ThErr = pthread_create(&ThHandle[i], &ThAttr, PThrThreshold, (void *)&ThParam[i]);
			if(ThErr != 0){
				printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
				exit(EXIT_FAILURE);
			}
			}


	
		checkCuda(launch_helper(CPU_InputArray, CPU_OutputArray, GPURuntimes), __LINE__); //checkcuda
						//joining CPU threads
		for(i=0; i<NumThreads; i++){
			pthread_join(ThHandle[i], NULL);
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


	//save image to disk
	Mat result = Mat(M,N,CV_8UC1, CPU_OutputArray);
	imwrite(filename,result);

		

		if (!imwrite(filename, result)){
			fprintf(stderr, "couldn't write output to disk!\n");
	cudaFreeHost(CPU_InputArray);
    cudaFreeHost(CPU_OutputArray);	

			exit(EXIT_FAILURE);
		} 
				
		printf("Saved image '%s', size = %dx%d (dims = %d).\n",
			   //filename.c_str(), result.cols, result.rows, result.dims);
			   filename, result.cols, result.rows, result.dims);
			   pthread_attr_destroy(&ThAttr);
  

  }
	cudaFreeHost(CPU_InputArray);
    cudaFreeHost(CPU_OutputArray);	 
	cudaFreeHost(CPU_Gauss);
	cudaFreeHost(CPU_Gradient);
	cudaFreeHost(CPU_Theta);
	cudaFreeHost(CPU_Threshold);
	//MISSING STUFF HERE
	cudaDeviceReset();

  // Done.
  exit(EXIT_SUCCESS);

} //end of main

// Helper function for launching the CUDA kernel (including memcpy, etc.):
cudaError_t launch_helper(uchar *CPU_InputArray, uchar *CPU_OutputArray, float *Runtimes){

		cudaEvent_t time1, time2, time3, time4;
		//int ucharGPUSize, doubleGPUSize; // total size of 1 image in bytes
		uchar *GPU_idata;
		uchar *GPU_odata;
		//uchar *GPU_zerodata;
		double *GPU_Gaussdata;
		double *GPU_Gradientdata;
		double *GPU_Thetadata;
		
		dim3 threadsPerBlock;
		dim3 numBlocks;

  	  dim3 sharedBlocks;
  	  dim3 streamSize;
  	  int UCharTotalSize = (M/levels+4)*N*sizeof(uchar);
	  int DoubleTotalGPUSize = M * N * sizeof(double);

  	threadsPerBlock = dim3(BOX_SIZE,BOX_SIZE);
	numBlocks = dim3(ceil((float)M / threadsPerBlock.x),ceil((float)N / threadsPerBlock.y));
	sharedBlocks = dim3(ceil((float)numBlocks.x/levels),ceil((float)numBlocks.y/stream));
	//shared_mem_size = threadsPerBlock.x*threadsPerBlock.y;
	printf("NumThreads/Block: %d, NumBlocks: %d, %d, Shared Blocks: %d, %d\n",threadsPerBlock.x*threadsPerBlock.y,numBlocks.x,numBlocks.y,sharedBlocks.x,sharedBlocks.y);


	cudaStream_t streams[stream];
	for (int i = 0; i < stream; ++i) { //one extra for dummy purposes
		checkCuda(cudaStreamCreate(&streams[i]), __LINE__); //checkcuda
	}
	
	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);
		

    // Allocate GPU buffer for input and output: all of them checkcuda
    cudaMalloc((void**)&GPU_idata, TotalGPUSize);
    cudaMalloc((void**)&GPU_odata, TotalGPUSize);
    cudaMalloc((void**)&GPU_Gaussdata, DoubleTotalGPUSize);
    cudaMalloc((void**)&GPU_Gradientdata, DoubleTotalGPUSize);
    cudaMalloc((void**)&GPU_Thetadata, DoubleTotalGPUSize);
	
    // Copy this frame to the GPU:
    cudaEventRecord(time1, 0);
    int offsetx, offsety;
	for (int i = 0; i < levels+1; i++) {
		if (i<levels) {
			if (i<levels-1) {
				//printf("\nCurrently on level: %d, Pinned memory offset: %d\n",i,TotalGPUSize/levels*i);
				checkCuda(cudaMemcpyAsync(&GPU_idata[TotalGPUSize/levels*i], &CPU_InputArray[TotalGPUSize/levels*i], UCharTotalSize, cudaMemcpyHostToDevice, streams[0]), __LINE__); //checkcuda
				
				offsetx = threadsPerBlock.x*sharedBlocks.x*i;
				for (int j = 0; j < stream; j++){
					offsety = j*sharedBlocks.y*threadsPerBlock.y;
					GaussianFilter<<<sharedBlocks, threadsPerBlock, 0 , streams[j]>>>(GPU_idata, GPU_Gaussdata, M, N, offsetx, offsety);  //CHECK THIS
					checkCuda( cudaGetLastError(), __LINE__ );
				}///////////////////////////////////////////
			}
			else if (i == levels - 1){
				checkCuda(cudaMemcpyAsync(&GPU_idata[TotalGPUSize/levels*i], &CPU_Threshold[TotalGPUSize/levels*i], TotalGPUSize/levels, cudaMemcpyHostToDevice, streams[0]), __LINE__); //checkcuda
			}
			cudaEventRecord(time2,0);
			// Launch kernel:
		}
		if (i>0) {
			if (i < levels){
				offsetx = threadsPerBlock.x*sharedBlocks.x*(i-1);
				for(int j = 0; j<stream; j++) {
					offsety = j*sharedBlocks.y*threadsPerBlock.y;
					Sobel<<<sharedBlocks, threadsPerBlock, 0 , streams[j]>>>(GPU_Gaussdata, GPU_Gradientdata, GPU_Thetadata, M, N, offsetx, offsety);
					checkCuda(cudaGetLastError(), __LINE__); //checkcuda
				}
				for(int j = 0; j<stream; j++) {
					offsety = j*sharedBlocks.y*threadsPerBlock.y;
					Threshold<<<sharedBlocks, threadsPerBlock, 0 , streams[j]>>>(GPU_Gradientdata, GPU_Thetadata, GPU_odata, M, N, offsetx, offsety, ThreshLo*8, ThreshLo*8 + 7);
					checkCuda(cudaGetLastError(), __LINE__); //checkcuda
				}
				cudaEventRecord(time3, 0);
				// Copy result back to CPU:
				//checkCuda( cudaMemcpyAsync(CPU_OutputArray, GPU_odata, TotalGPUSize,
				//			   cudaMemcpyDeviceToHost, streams[0]) );
				checkCuda(cudaMemcpyAsync(&CPU_OutputArray[TotalGPUSize/levels*(i-1)], &GPU_odata[TotalGPUSize/levels*(i-1)], TotalGPUSize/levels, cudaMemcpyDeviceToHost, streams[0]), __LINE__); //checkcuda
			}
			else {//last block --> assigned to CPU
				checkCuda( cudaMemcpyAsync(&CPU_OutputArray[TotalGPUSize/levels*(i-1)], &CPU_Threshold[TotalGPUSize/levels*(i-1)], TotalGPUSize/levels, cudaMemcpyDeviceToHost, streams[0]), __LINE__ );
			}



		}
	} //end of level loop

/**	pthread_attr_init(&ThAttr);
	pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);
	for(int i=0; i<NumThreads; i++){

		ThParam[i] = i;

		ThErr = pthread_create(&ThHandle[i], &ThAttr, _________, (void *)&ThParam[i]);

	//printf("Successfully created thread!\n");
		if(ThErr != 0){
			printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
			exit(EXIT_FAILURE);
		}
		  for(int i=0; i<NumThreads; i++){
			pthread_join(ThHandle[i], NULL);
		}


	}**/
	
		cudaEventRecord(time4, 0);
  // cudaDeviceSynchronize waits for all preceding tasks to finish, and returns
  // an error if any of them failed:
  checkCuda(cudaDeviceSynchronize(), __LINE__); //checkcuda

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

  // Clean up memory:
  for (int i = 0; i < stream; ++i) {
  			checkCuda(cudaStreamDestroy(streams[i]), __LINE__); //checkcuda
  		  }

    cudaFree(GPU_odata);
    cudaFree(GPU_idata);
    cudaFree(GPU_Gaussdata);
    cudaFree(GPU_Gradientdata);
    cudaFree(GPU_Thetadata);
  //free(GPU_odata);
  //free(GPU_idata);
    cudaEventDestroy(time1);
    	cudaEventDestroy(time2);
    	cudaEventDestroy(time3);
    	cudaEventDestroy(time4);


	ThreshLo++;
	// Done.
	return cudaSuccess;
}
