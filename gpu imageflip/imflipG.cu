// ECE 406 Lab 5, Fall 2015

#include <stdio.h>

// CUDA stuff:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// OpenCV stuff (note: C++ not C):
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes);

int M;  // number of rows in image
int N;  // number of columns in image

// These come from CLI arguments:
int BOX_SIZE;           // ThreadsPerBlock == BOX_SIZE*BOX_SIZE
bool show_images;	// whether we should pop up the I/O images or not

__global__ void lab5_kernel(uchar *GPU_i, uchar *GPU_o, int M, int N)
{
	//printf("int blockIdx.x = %d and int blockIdx.y = %d\n", blockIdx.x, blockIdx.y);
	/*if (blockDim.x != 2 && blockDim.y != 2){
	printf("int blockDim.x = %d and int blockDim.y = %d\n", blockDim.x, blockDim.y);
	}*/
	//printf("int threadIdx.x = %d and int threadIdx.y = %d\n", threadIdx.x, threadIdx.y);
	//printf("M = %d N = %d\n", M, N);
	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // column of image
	int idx = i*N + j;  // which pixel in full 1D array
	int i2 = i;  // new row of image
	int j2 = N - j;  // new column of image
	int idx2 = i2*N + j2; 

	uchar output = GPU_i[idx];  // no change, REPLACE THIS
		//printf("idx = %d = %d * %d + %d\n", idx, i, N, j);
	if (i <= 10 && j <= 10){
	//printf("idx = %d = %d * %d + %d\n", idx, i, N, j);
	//printf("GPU_i [%d] = %u\n", idx, GPU_i[idx]);
	}
	GPU_o[idx2] = output;
}

// Display image until a key is pressed in the window:
void show_image(Mat image, string title) {
	if (show_images) {
		namedWindow(title, WINDOW_AUTOSIZE);  // create window
		imshow(title, image);                 // show image
		waitKey(0);
	}
}

int main(int argc, char *argv[])
{
	float GPURuntimes[4];		// run times of the GPU code
	cudaError_t cudaStatus;
	int *CPU_OutputArray;		// where the GPU should copy the output back to

	if( argc != 5) {
	  printf("Usage: %s <input image> <output image> <box size> <show images>\n", argv[0]);
		printf("       where 'show images' is 0 or 1\n");
		exit(EXIT_FAILURE);
	}
	BOX_SIZE = atoi(argv[3]);
	show_images = atoi(argv[4]);

	// Load image:
	Mat image;	// see http://docs.opencv.org/modules/core/doc/basic_structures.html#mat
	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	// we could load it as CV_LOAD_IMAGE_COLOR, but we don't want to worry about that extra dimension
	if(! image.data ) {
		fprintf(stderr, "Could not open or find the image.\n");
		exit(EXIT_FAILURE);
	}
	printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);

	// Set up global variables based on image size:
	M = image.rows;
	N = image.cols;

	// Display the input image:
	show_image(image, "input image");

	// Create CPU memory to store the output:
	CPU_OutputArray = (int*)malloc(M*N*sizeof(int));
	if (CPU_OutputArray == NULL) {
		fprintf(stderr, "OOPS. Can't create CPU_OutputArray using malloc() ...\n");
		exit(EXIT_FAILURE);
	}

	// Run it:
	cudaStatus = launch_helper(image, CPU_OutputArray, GPURuntimes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "launch_helper failed!\n");
		free(CPU_OutputArray);
		exit(EXIT_FAILURE);
	}

	printf("-----------------------------------------------------------------\n");
	printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \n Total=%5.2f ms\n",
			GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
	printf("-----------------------------------------------------------------\n");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		free(CPU_OutputArray);
		exit(EXIT_FAILURE);
	}

	// Display the output image:
	Mat result = Mat(M, N, CV_8UC1, CPU_OutputArray);
	show_image(result, "output image");
	// and save it to disk:
	string output_filename = argv[2];
	if (!imwrite(output_filename, result)) {
		fprintf(stderr, "couldn't write output to disk!\n");
		free(CPU_OutputArray);
		exit(EXIT_FAILURE);
	}
	printf("Saved image '%s', size = %dx%d (dims = %d).\n",
	       output_filename.c_str(), result.rows, result.cols, result.dims);

	free(CPU_OutputArray);
	exit(EXIT_SUCCESS);
}

// Helper function for launching a CUDA kernel (including memcpy, timing, etc.):
cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes)
{
	cudaEvent_t time1, time2, time3, time4;
	int TotalGPUSize;  // total size of 1 image (i.e. input or output) in bytes
	uchar *GPU_idata;
	uchar *GPU_odata;
	// Note that we could store GPU_i and GPU_o as 2D arrays instead of 1D...
	// it would make indexing simpler, but could complicate memcpy.

	dim3 threadsPerBlock;
	dim3 numBlocks;

	// Choose which GPU to run on; change this on a multi-GPU system.
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);  // use the first GPU (not necessarily the fastest)
	if (cudaStatus != cudaSuccess) {
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

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPU_idata, image.data, TotalGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaEventRecord(time2, 0);

	// Launch a kernel on the GPU with one thread for each pixel.
	threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
	numBlocks = dim3(M / threadsPerBlock.x, N / threadsPerBlock.y);
	lab5_kernel<<<numBlocks, threadsPerBlock>>>(GPU_idata, GPU_odata, M, N);

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

	return cudaStatus;
}
