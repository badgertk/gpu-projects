// ECE 406 Lab 8, Fall 2015

#include <stdio.h>
#include <assert.h>

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

cudaError_t launch_helper(uchar **CPU_InputArray, uchar **CPU_OutputArray);

int M;  // number of rows in image
int N;  // number of columns in image
int TotalSize;  // total size of image in bytes

// These come from command line arguments:
int threadsPerBlock;
bool show_images;	// whether we should pop up the I/O images or not
int nframes;            // we will 'replay' the single image nframes times

// Flip image horizontally.  Each thread handles one row:
__global__ void lab8_kernel(uchar *GPU_i, uchar *GPU_o, int M, int N)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) { return; }  // do nothing if I'm outside of the image

  int start = row*N*3;        // first byte of row in 1D array
  int end = start + N*3 - 1;  // last byte of row in 1D array

  int col;
  for (col=0; col<N*3; col+=3) {
    GPU_o[start+col]   = GPU_i[end-col-2];
    GPU_o[start+col+1] = GPU_i[end-col-1];
    GPU_o[start+col+2] = GPU_i[end-col];
  }
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
  // Parse input args:
  if ( argc != 6 ) {
    printf("Usage: %s <input image> <output image> <block size> <nframes> <show images>\n",
	   argv[0]);
    printf("       where 'show images' is 0 or 1\n");
    exit(EXIT_FAILURE);
  }
  threadsPerBlock = atoi( argv[3] );
  nframes         = atoi( argv[4] );
  show_images     = atoi( argv[5] );

  // where the GPU should copy the data from/to:
  uchar **CPU_InputArray  = (uchar **)malloc(nframes * sizeof(uchar*));
  uchar **CPU_OutputArray = (uchar **)malloc(nframes * sizeof(uchar*));
  if ((CPU_InputArray == NULL) || (CPU_OutputArray == NULL)) {
    fprintf(stderr, "OOPS. Can't create I/O array(s) using malloc() ...\n");
    exit(EXIT_FAILURE);
  }

  // Load input image:
  Mat image;	// see http://docs.opencv.org/modules/core/doc/basic_structures.html#mat
  image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if(! image.data ) {
    fprintf(stderr, "Could not open or find the image.\n");
    exit(EXIT_FAILURE);
  }
  printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);

  // Set up global variables based on image size:
  M = image.rows;
  N = image.cols;
  TotalSize = M * N * sizeof(uchar) * 3;

  // Display the input image:
  show_image(image, "input image");

  // Copy the image to the input array.  We'll duplicate it nframes times.
  int i;
  for (i=0; i<nframes; i++) {
    checkCuda( cudaMallocHost( (void**)&CPU_InputArray[i],  TotalSize ) );
    memcpy(CPU_InputArray[i], image.data, TotalSize);  // always the same image
    //  Allocate the output while we're at it:
    checkCuda( cudaMallocHost( (void**)&CPU_OutputArray[i], TotalSize ) );
  }
  
  // Run it:
  checkCuda( launch_helper(CPU_InputArray, CPU_OutputArray) );

  // Display the (last) output image:
  Mat result = Mat(M, N, CV_8UC3, CPU_OutputArray[nframes-1]);
  show_image(result, "output image");
  // and save it to disk:
  string output_filename = argv[2];
  if (!imwrite(output_filename, result)) {
    fprintf(stderr, "couldn't write output to disk!\n");
    exit(EXIT_FAILURE);
  }
  printf("Saved image '%s', size = %dx%d (dims = %d).\n",
	 output_filename.c_str(), result.rows, result.cols, result.dims);

  // Clean up memory:
  for (i=0; i<nframes; i++) {
    cudaFreeHost(CPU_InputArray[i]);
    cudaFreeHost(CPU_OutputArray[i]);
  }
  free(CPU_InputArray);
  free(CPU_OutputArray);

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Parallel Nsight and Visual Profiler to show complete
  // traces.  Don't call it before you're done using the pinned memory!
  checkCuda( cudaDeviceReset() );

  // Done.
  exit(EXIT_SUCCESS);
}

// Helper function for launching the CUDA kernel (including memcpy, etc.):
cudaError_t launch_helper(uchar **CPU_InputArray, uchar **CPU_OutputArray)
{
  // pointers to GPU data arrays:
  uchar **GPU_idata = (uchar **)malloc(nframes * sizeof(uchar*));
  uchar **GPU_odata = (uchar **)malloc(nframes * sizeof(uchar*));
  if ((GPU_idata == NULL) || (GPU_odata == NULL)) {
    fprintf(stderr, "OOPS. Can't create GPU I/O array(s) using malloc() ...\n");
    return(cudaErrorUnknown);  // could do cudaErrorMemoryAllocation, but we're
			       // not really here due to a CUDA error
  }

  // Number of blocks is ceil(M/threadsPerBlock), same for every block:
  int numBlocks = (M % threadsPerBlock) ?
    M / threadsPerBlock + 1 :
    M / threadsPerBlock;

  // Loop over frames:
  int i;
  for (i = 0; i < nframes; i++) {
    // Allocate GPU buffer for input and output:
    checkCuda( cudaMalloc((void**)&GPU_idata[i], TotalSize) );
    checkCuda( cudaMalloc((void**)&GPU_odata[i], TotalSize) );

    // Copy this frame to the GPU:
    checkCuda( cudaMemcpyAsync(GPU_idata[i], CPU_InputArray[i], TotalSize,
			       cudaMemcpyHostToDevice) );

    // Launch kernel:
    lab8_kernel<<<numBlocks, threadsPerBlock>>>(GPU_idata[i], GPU_odata[i], M, N);
    checkCuda( cudaGetLastError() );

    // Copy result back to CPU:
    checkCuda( cudaMemcpyAsync(CPU_OutputArray[i], GPU_odata[i], TotalSize,
			       cudaMemcpyDeviceToHost) );
  }

  // cudaDeviceSynchronize waits for all preceding tasks to finish, and returns
  // an error if any of them failed:
  checkCuda( cudaDeviceSynchronize() );

  // Clean up memory:
  for (i = 0; i < nframes; i++) {
    cudaFree(GPU_odata[i]);
    cudaFree(GPU_idata[i]);
  }
  free(GPU_odata);
  free(GPU_idata);

  // Done.
  return cudaSuccess;
}
