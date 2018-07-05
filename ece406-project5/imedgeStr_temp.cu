#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>

// CUDA stuff:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// OpenCV stuff (note: C++ not C):
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv; 

// Convenience function for checking CUDA runtime API results can be wrapped
// around any runtime API call. Source:
// https://github.com/parallel-forall/code-samples
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
#define EDGE		 255
#define NOEDGE       0
cudaError_t launch_helper(uchar *CPU_InputArray, uchar *CPU_OutputArray, float *GPURuntimes);

long  			NumThreads;         		// Total number of threads working in parallel
//int 	     	ThParam[MAXTHREADS];		// Thread parameters ...
//double			RotAngle;					// rotation angle
//void* (*RotateFunc)(void *arg);				// Function pointer to rotate the image (multi-threaded)
//int nframes;
//int BOX_SIZE;
//int version;
int M;  // number of rows in image
int N;  // number of columns in image
int TotalSize;
int TotalSize_2;

int nStreams = 4;
int levels = 8;

int Thresh;

int BOX_SIZE;           // ThreadsPerBlock == BOX_SIZE*BOX_SIZE
bool show_images;	// whether we should pop up the I/O images or not

uchar *CPU_InputArray;
uchar *CPU_OutputArray;

// Sobel kernels:
float H[3][3] = { { -1,  0,  1 },
		  { -2,  0,  2 },
		  { -1,  0,  1 } };
float V[3][3] = { { -1, -2, -1 },
		  {  0,  0,  0 },
		  {  1,  2,  1 } };

__device__ double Gx[3][3] = {		{ -1, 0, 1 },
						{ -2, 0, 2 },
						{ -1, 0, 1 }	};

__device__ double Gy[3][3] = {		{ -1, -2, -1 },
						{  0,  0,  0 },
						{  1,  2,  1 }	};

__device__ double Gauss[5][5] = {	{ 2, 4,  5,  4,  2 },
						{ 4, 9,  12, 9,  4 },
						{ 5, 12, 15, 12, 5 },
						{ 4, 9,  12, 9,  4 },
						{ 2, 4,  5,  4,  2 }	};


// Function that takes BWImage and calculates the Gaussian filtered version
// Saves the result in the GaussFilter[][] array
__global__ void GaussianFilter(uchar *GPU_i, uchar *GPU_o, int M, int N, int offsetx, int offsety)
{
    //extern __shared__ uchar GPU_i_shared[];
	//long tn;            // My thread number (ID) is stored here
    int i,j;
	double G;  			// temp to calculate the Gaussian filtered version

	//__shared__ double Gauss[25];

	/*Gauss = {	 2, 4,  5,  4,  2 ,
									 4, 9,  12, 9,  4 ,
									 5, 12, 15, 12, 5 ,
									 4, 9,  12, 9,  4 ,
									 2, 4,  5,  4,  2 	};*/
    //tn = *((int *) tid);           // Calculate my Thread ID
    //tn *= ip.Vpixels/NumThreads;
	int rt = blockIdx.x * blockDim.x + threadIdx.x+offsetx;  // row of image
	int ct = blockIdx.y * blockDim.y + threadIdx.y+offsety;  // column of image
	//int k;
	int idx = rt*N+ct;  // which pixel in full 1D array
	int idy;
	//int idz = threadIdx.x*blockDim.x+threadIdx.y;
	if (rt<M&&ct<N) {
		//GPU_i_shared[idz] = GPU_i[idx];
		//printf("IDX : %d*%d+%d = %d , IDZ : %d\n",rt,N,ct,idx,idz);
		//for(row=tn; row<tn+M/NumThreads; row++)
		//{
		//if (rt>=M || ct>=N) return;
		//__syncthreads();
		if((rt>1) && (rt<(M-2))) {
		//col=2;
			if(ct<(N-2)&&ct>1){
				G=0.0;
				for(i=-2; i<=2; i++){
					for(j=-2; j<=2; j++){
						idy = (rt+i)*N+ct+j;
						//idy = (threadIdx.x+i)*blockDim.x+threadIdx.y+j;
						G+=GPU_i[idy]*Gauss[i+2][j+2];
						//printf("Gauss: %10.4f, GPU_i: %d, G: %10.4f\n",Gauss[i+2][j+2],GPU_i[idy],G);
					}
				}
				GPU_o[idx]=G/159.000;
				//col++;
				//printf("GPU_o %d : %d\n",idx,GPU_o[idx]);
			}
		}
	}
	//else GPU_o[idx] = 0;
    //}
    //pthread_exit(NULL);
}


// Function that calculates the Gradient and Theta for each pixel
// Takes the Gauss[][] array and creates the Gradient[][] and Theta[][] arrays
__global__ void Sobel(uchar *GPU_i, uchar *Gradient, int M, int N, int offsetx, int offsety)
{
    //long tn;            		     // My thread number (ID) is stored here
    int i,j;
	double GX,GY;

    int rt = blockIdx.x * blockDim.x + threadIdx.x+offsetx;  // row of image
	int ct = blockIdx.y * blockDim.y + threadIdx.y+offsety;  // column of image
	//int k;
	int idx = rt*N+ct;  // which pixel in full 1D array
	int idy;
    /*for(row=tn; row<tn+ip.Vpixels/NumThreads; row++)
    {
		if((row<1) || (row>(ip.Vpixels-2))) continue;
        col=1;
        while(col<=(N-2)){*/
	if (rt<M&&ct<N) {
		if((rt>0) && (rt<(M-1))) {

			if(ct<=(N-2)&&ct>0){
				// calculate Gx and Gy
				GX=0.0; GY=0.0;
				for(i=-1; i<=1; i++){
					for(j=-1; j<=1; j++){
						idy = (rt+i)*N+ct+j;
						GX+=GPU_i[idy]*Gx[i+1][j+1];
						GY+=GPU_i[idy]*Gy[i+1][j+1];
					}
				}
				//if (rt == 5) printf("G = %f\n",sqrt(GX*GX+GY*GY));
				Gradient[idx]=sqrt(GX*GX+GY*GY);
				//Theta[idx]=atan(GX/GY)*180.0/PI;
				//col++;
			}
		}
	}
	else return;
    //pthread_exit(NULL);
}

__global__ void Threshold(uchar *GPU_i, uchar *GPU_o,  int M, int N, int Thresh, int offsetx, int offsety)
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
    //for(row=tn; row<tn+ip.Vpixels/NumThreads; row++)
    //{
		//if((row<1) || (row>(M-2))) continue;
        //col=1;
	//if(ct>0 && ct<=(N-2)){
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
		//CopyImage[row][col*3+1]=PIXVAL;
		//CopyImage[row][col*3+2]=PIXVAL;
		//col++;
	//}
}
    //pthread_exit(NULL);





int main(int argc, char *argv[])
{
	float GPURuntimes[4];
	float GPUexetime = 0.0;
  // Parse input args:
  if ( argc != 3 ) {
    printf("Usage: %s <input output> <image> \n",
	   argv[0]);
    printf("       where 'show images' is 0 or 1\n");
    exit(EXIT_FAILURE);
  }
  BOX_SIZE = 16;
  //show_images     = atoi( argv[5] );
  Thresh = 0;
  int j = 1;
  // where the GPU should copy the data from/to:

  /*if ((CPU_InputArray == NULL) || (CPU_OutputArray == NULL)) {
    fprintf(stderr, "OOPS. Can't create I/O array(s) using malloc() ...\n");
    exit(EXIT_FAILURE);
  }*/

  // Load input image:
  Mat image;	// see http://docs.opencv.org/modules/core/doc/basic_structures.html#mat
  image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  if(! image.data ) {
    fprintf(stderr, "Could not open or find the image.\n");
    exit(EXIT_FAILURE);
  }
  printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);

  // Set up global variables based on image size:
  M = image.rows;
  N = image.cols;
  TotalSize = M * N * sizeof(uchar);

  // Display the input image:
  //show_image(image, "input image");

  // Copy the image to the input array.  We'll duplicate it nframes times.
  //int i;
  //for (i=0; i<nframes; i++) {
    checkCuda( cudaMallocHost( (void**)&CPU_InputArray,  TotalSize ) );
    memcpy(CPU_InputArray, image.data, TotalSize);  // always the same image
    //  Allocate the output while we're at it:
    checkCuda( cudaMallocHost( (void**)&CPU_OutputArray, TotalSize ) );
  //}
  
  for (Thresh=0;Thresh<=128;Thresh+=8) {




  // Run it:
  checkCuda( launch_helper(CPU_InputArray, CPU_OutputArray, GPURuntimes) );

  	printf("-----------------------------------------------------------------\n");
	printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \n Total=%5.2f ms\n",
			GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
	printf("-----------------------------------------------------------------\n");

	GPUexetime += GPURuntimes[2];
	// Display the (last) output image:
	Mat result = Mat(M, N, CV_8UC1, CPU_OutputArray);
	//show_image(result, "output image");
	// and save it to disk:
	string output_filename = argv[2];
	//printf("i : %d\n",i);
	char n0, n1;
	if (j>9) {
		n0 = '1';
		n1 = (j-10)+'0';
		output_filename.insert(output_filename.end()-4,n0);
		output_filename.insert(output_filename.end()-4,n1);
	}

	else {
		n0 = j+'0';
		output_filename.insert(output_filename.end()-4,n0);
	}

  if (!imwrite(output_filename, result)) {
    fprintf(stderr, "couldn't write output to disk!\n");
    exit(EXIT_FAILURE);
  }
  printf("Saved image '%s', size = %dx%d (dims = %d).\n",
	 output_filename.c_str(), result.rows, result.cols, result.dims);
  j++;
  }
  // Clean up memory:
    cudaFreeHost(CPU_InputArray);
    cudaFreeHost(CPU_OutputArray);
  //free(CPU_InputArray);
  //free(CPU_OutputArray);

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Parallel Nsight and Visual Profiler to show complete
  // traces.  Don't call it before you're done using the pinned memory!
  checkCuda( cudaDeviceReset() );

  // Done.
  exit(EXIT_SUCCESS);
}

// Helper function for launching the CUDA kernel (including memcpy, etc.):
cudaError_t launch_helper(uchar *CPU_InputArray, uchar *CPU_OutputArray, float *Runtimes)
{
	cudaEvent_t time1, time2, time3, time4;
  // pointers to GPU data arrays:
  uchar *GPU_idata;
  uchar *GPU_odata;
  uchar *GaussImage;
  //	uchar *Gradient;
  /*if ((GPU_idata == NULL) || (GPU_odata == NULL)) {
    fprintf(stderr, "OOPS. Can't create GPU I/O array(s) using malloc() ...\n");
    return(cudaErrorUnknown);  // could do cudaErrorMemoryAllocation, but we're
			       // not really here due to a CUDA error
  }*/

  // Number of blocks is ceil(M/threadsPerBlock), same for every block:
  	  dim3 threadsPerBlock;
  	  dim3 numBlocks;
  	  dim3 sharedBlocks;
  	  int shared_mem_size;
  	  dim3 streamSize;
  	  TotalSize_2 = (M/levels+4)*N*sizeof(uchar);

  	threadsPerBlock = dim3(BOX_SIZE,BOX_SIZE);
	numBlocks = dim3(ceil((float)M / threadsPerBlock.x),ceil((float)N / threadsPerBlock.y));
	sharedBlocks = dim3(ceil((float)numBlocks.x/levels),ceil((float)numBlocks.y/nStreams));
	shared_mem_size = threadsPerBlock.x*threadsPerBlock.y;
	printf("NumThreads/Block: %d, NumBlocks: %d, %d, Shared Blocks: %d, %d\n",threadsPerBlock.x*threadsPerBlock.y,numBlocks.x,numBlocks.y,sharedBlocks.x,sharedBlocks.y);


	cudaStream_t stream[nStreams+1];
	//checkCuda( cudaEventCreate(&startEvent) );
	//checkCuda( cudaEventCreate(&stopEvent) );
	for (int i = 0; i < nStreams+1; ++i) {
		checkCuda( cudaStreamCreate(&stream[i]) );
	}
	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);
	//printf("TotalSize = %d\n",TotalSize);
	//printf("TotalSize_2 = %d\n",TotalSize_2);

  // Loop over frames:

    // Allocate GPU buffer for input and output:
    checkCuda( cudaMalloc((void**)&GPU_idata, TotalSize) );
    checkCuda( cudaMalloc((void**)&GPU_odata, TotalSize) );
    checkCuda( cudaMalloc((void**)&GaussImage, TotalSize) );

    // Copy this frame to the GPU:
    cudaEventRecord(time1, 0);
    int offsetx, offsety;
	for (int i = 0; i < levels+1; i++) {
		if (i<levels) {
		if (i<levels-1) {
			//printf("\nCurrently on level: %d, Pinned memory offset: %d\n",i,TotalSize/levels*i);
			checkCuda( cudaMemcpyAsync(&GPU_idata[TotalSize/levels*i], &CPU_InputArray[TotalSize/levels*i], TotalSize_2,
								   cudaMemcpyHostToDevice, stream[0]) );
		}
		else if (i==levels-1) {
			checkCuda( cudaMemcpyAsync(&GPU_idata[TotalSize/levels*i], &CPU_InputArray[TotalSize/levels*i], TotalSize/levels,
						   	   	   cudaMemcpyHostToDevice, stream[0]) );
		}
		cudaEventRecord(time2,0);
		// Launch kernel:

		offsetx = threadsPerBlock.x*sharedBlocks.x*i;
		for(int j = 0; j<nStreams; j++) {

			offsety = j*sharedBlocks.y*threadsPerBlock.y;
			GaussianFilter<<<sharedBlocks, threadsPerBlock, 0 , stream[j+1]>>>(GPU_idata, GPU_odata, M, N, offsetx, offsety);
			checkCuda( cudaGetLastError() );
		}
		}
		if (i>0) {
			offsetx = threadsPerBlock.x*sharedBlocks.x*(i-1);
		for(int j = 0; j<nStreams; j++) {
			offsety = j*sharedBlocks.y*threadsPerBlock.y;
			Sobel<<<sharedBlocks, threadsPerBlock, 0 , stream[j+1]>>>(GPU_odata, GaussImage, M, N, offsetx, offsety);
			checkCuda( cudaGetLastError() );
		}
		for(int j = 0; j<nStreams; j++) {
			offsety = j*sharedBlocks.y*threadsPerBlock.y;
			Threshold<<<sharedBlocks, threadsPerBlock, 0 , stream[j+1]>>>(GaussImage, GPU_odata, M, N, Thresh, offsetx, offsety);
			checkCuda( cudaGetLastError() );
		}

		cudaEventRecord(time3, 0);
		// Copy result back to CPU:
		//checkCuda( cudaMemcpyAsync(CPU_OutputArray, GPU_odata, TotalSize,
		//			   cudaMemcpyDeviceToHost, stream[0]) );
		checkCuda( cudaMemcpyAsync(&CPU_OutputArray[TotalSize/levels*(i-1)], &GPU_odata[TotalSize/levels*(i-1)], TotalSize/levels,
							   cudaMemcpyDeviceToHost, stream[0]) );
		cudaEventRecord(time4, 0);
		}
	}

  // cudaDeviceSynchronize waits for all preceding tasks to finish, and returns
  // an error if any of them failed:
  checkCuda( cudaDeviceSynchronize() );

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
  for (int i = 0; i < nStreams+1; ++i) {
  			checkCuda( cudaStreamDestroy(stream[i]) );
  		  }

    cudaFree(GPU_odata);
    cudaFree(GPU_idata);
    cudaFree(GaussImage);
  //free(GPU_odata);
  //free(GPU_idata);
    cudaEventDestroy(time1);
    	cudaEventDestroy(time2);
    	cudaEventDestroy(time3);
    	cudaEventDestroy(time4);

  // Done.
  return cudaSuccess;
}
