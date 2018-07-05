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

cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes);

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

#define BOX_SIZE		16 		//ThreadsPerBlock == BOX_SIZE * BOX_SIZE
#define PI				3.1415926
#define EDGE			255
#define NOEDGE			0
#define SHARED_MEM_LIMIT 48*1024

int M; //number of rows in the image
int N; //number of columns in the image

char Type;
int Xo; // Width of structuring element
int Yo; // Height of structuring element
uchar * StrElement;

int NumIter = 16;
int Thresh = 32;
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

__global__ void Erosion(uchar *GPU_i, uchar *Erosiondata, uchar *StrElement, int M, int N, int Xo, int Yo){

	extern __shared__ uchar shared_GPU_i[]; //does this get initialized at 0?  if not need to add 0 conditions to rest of conditionals
	
	int row = blockIdx.x * blockDim.x + threadIdx.x; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	int idy;
	int ids = (threadIdx.x+Yo/2)*blockDim.y+threadIdx.y+Xo/2;
	//uchar output = GPU_i[idx];
	//uchar ElementResult;

	int min;
	int i,j;
	int d = Yo/2;
	int e = Xo/2;
	min = 255;
	//printf("gridDim.x: %d \ngridDim.y: %d\n",gridDim.x,gridDim.y);
	if (row >= M || col >= N ) goto End;

	if ((blockIdx.x > 0) && (blockIdx.x < gridDim.x-1)){//if not first row or last row
		if ((blockIdx.y > 0) && (blockIdx.y < gridDim.y-1)){//if not first column or last column (general case)
			if ((threadIdx.x == 0) && (threadIdx.y == 0)){ //top left of block
				for (i=-d; i<=0; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == 0) && (threadIdx.y == (blockDim.y - 1))){ //top right
				for (i=-d; i<=0; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == 0)){ //bottom left
				for (i=0; i<=d; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == (blockDim.y - 1))){ //bottom right
				for (i=0; i<=d; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x < blockDim.x-1) && (threadIdx.y == 0)) { //left edge
				for (j=-e; j<=0; j++) {
					ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
					idy = (row)*N+col+j;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x < blockDim.x-1) && (threadIdx.y == blockDim.y-1)) { // Right edge
				for (j=0; j<=e; j++) {
					ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
					idy = (row)*N+col+j;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.y > 0) && (threadIdx.y < blockDim.y-1) && (threadIdx.x == 0)) { //top edge
				for (i=-d; i<=0; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.y > 0) && (threadIdx.y < blockDim.y-1) && (threadIdx.x == blockDim.x-1)) { //Bottom edge
				for (i=0; i<=d; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		if (blockIdx.y == 0){//leftmost column - left edge
			if ((threadIdx.x == 0) && (threadIdx.y == (blockDim.y - 1))){ //top right
				for (i=-d; i<=0; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == (blockDim.y - 1))){ //bottom right
				for (i=0; i<=d; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x < blockDim.x-1) && (threadIdx.y == blockDim.y-1)) { //right edge
				for (j=0; j<=e; j++) {
					ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j;
					idy = (row)*N+col+j;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.y < blockDim.x-1) && (threadIdx.x == 0)) { //top edge
				for (i=-d; i<=0; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.y < blockDim.x-1) && (threadIdx.x == blockDim.x-1)) { //Bottom edge
				for (i=0; i<=d; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		if (blockIdx.y == (gridDim.y - 1)){//if last column - right edge
			if ((threadIdx.x == 0) && (threadIdx.y == 0)){ //top left of block
				for (i=-d; i<=0; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == 0)){ //bottom left
				for (i=0; i<=d; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x < blockDim.x-1) && (threadIdx.y == 0)) {//left edge
				for (j=-e; j<=0; j++) {
					ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
					idy = (row)*N+col+j;
					shared_GPU_i[ids]= GPU_i[idy];
				}	
			}
			if ((threadIdx.y > 0) && (threadIdx.y <= blockDim.y-1) && (threadIdx.x == 0)) { //top edge
				for (i=-d; i<=0; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.y > 0) && (threadIdx.y <= blockDim.y-1) && (threadIdx.x == blockDim.x-1)) { //Bottom edge
				for (i=0; i<=d; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
	}
	
	if (blockIdx.x == 0){//if first row
		
		if ((blockIdx.y > 0) && (blockIdx.y < gridDim.y-1)){//if not first column or last column - top edge
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == 0)){ //bottom left
				for (i=0; i<=d; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == (blockDim.y - 1))){ //bottom right
				for (i=0; i<=d; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x < blockDim.x-1) && (threadIdx.y == 0)) {// Left edge

					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.x < blockDim.x-1) && (threadIdx.y == blockDim.y-1) ){ // Right edge
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.y > 0) && (threadIdx.y < blockDim.y-1) && (threadIdx.x == blockDim.x-1)) { //bottom edge

				for (i=0; i<=d; i++) {
					ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+e;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			else { //everything else
				ids = (threadIdx.x)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		if (blockIdx.y == 0){//if first column - top left corner

			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == (blockDim.y - 1))){ //bottom right
				for (i=0; i<=d; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+j;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x < blockDim.y-1) && (threadIdx.y == blockDim.y-1)) { //right edge
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x)*blockDim.y+threadIdx.y+j;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.y < blockDim.y-1) && (threadIdx.x == blockDim.x-1)) { //bottom edge
					for (i=0; i<=d; i++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y;
						idy = (row+i)*N+col;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			else { //everything else
				ids = (threadIdx.x)*blockDim.y+threadIdx.y;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
				
		if (blockIdx.y == (gridDim.y - 1)){//if last column - top right corner

			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == 0)){ //bottom left
				for (i=0; i<=d; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			
			if ((threadIdx.x < blockDim.x-1) && (threadIdx.y == 0)) { // Left edge
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				
			}
			if ((threadIdx.y > 0) && (threadIdx.y <= blockDim.y-1) && (threadIdx.x == blockDim.x-1) ){ //Bottom edge

					for (i=0; i<=d; i++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+e;
						idy = (row+i)*N+col;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			else { //everything else
				ids = (threadIdx.x)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
	}
	
	
	if (blockIdx.x == (gridDim.x - 1)){//if last row
		if ((blockIdx.y > 0) && (blockIdx.y < gridDim.y-1)){//if not first column or last column
			if ((threadIdx.x == 0) && (threadIdx.y == 0)){ //top left of block
				for (i=-d; i<=0; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == 0) && (threadIdx.y == (blockDim.y - 1))){ //top right
				for (i=-d; i<=0; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x <= blockDim.x-1) && (threadIdx.y == 0)) { // Left edge

					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.x > 0) && (threadIdx.x <= blockDim.x-1) && (threadIdx.y == blockDim.y-1) ){ // Right edge
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.y > 0) && (threadIdx.y < blockDim.y-1) && (threadIdx.x == 0)) { // Top edge
					for (i=-d; i<=0; i++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
						idy = (row+i)*N+col;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		
		if (blockIdx.y == 0){//if first column - bottom left corner
			if ((threadIdx.x == 0) && (threadIdx.y == (blockDim.y - 1))){ //top right
				for (i=-d; i<=0; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}

			if ((threadIdx.x > 0) && (threadIdx.x <= blockDim.x-1) && (threadIdx.y == blockDim.y-1)) { // Right edge
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.y < blockDim.x-1) && (threadIdx.x == 0)) { // Top edge
					for (i=-d; i<=0; i++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y;
						idy = (row+i)*N+col;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		if (blockIdx.y == (gridDim.y - 1)){//if last vertical block - bottom right corner
			if ((threadIdx.x == 0) && (threadIdx.y == 0)){ //top left of block
				for (i=-d; i<=0; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			
			if ((threadIdx.x > 0) && (threadIdx.x <= blockDim.x-1) && (threadIdx.y == 0)) { // Left edge
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.y > 0) && (threadIdx.y <= blockDim.y-1) && (threadIdx.x == 0)) { // Top edge
					for (i=-d; i<=0; i++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
						idy = (row+i)*N+col;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
	}
	
	/*if ((threadIdx.x<(blockDim.x-1)) && (threadIdx.y<(blockDim.y-1)))
		shared_GPU_i[ids]= GPU_i[idx];
	else {
		if (threadIdx.x==(blockDim.x-1) && (threadIdx.y<(blockDim.y-1))) {
			for (i=0; i<=d; i++) {
				ids = (threadIdx.x+i)*blockDim.y+threadIdx.y;
				idy = (row+i)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		if (threadIdx.x<(blockDim.x-1) && (threadIdx.y==(blockDim.y-1))) {
			for (j=0; j<=e; j++) {
				ids = (threadIdx.x)*blockDim.y+threadIdx.y+j;
				idy = (row)*N+col+j+1;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		else {
			for (i=0; i<=d; i++) {
				for (j=0; j<=e; j++) {
					ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+j;
					idy = (row+i)*N+col+j;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
		}

	}**/
	//printf("ids : %d\n",ids);
	
	__syncthreads();

	
	/**if ((row < d) || (row > (M - d-1))) goto End;
	int idx2, row2, col2;

	for (i=-d; i<=d; i++){
		for (j=-e; j<=e; j++){
			if (min == 0) continue;
				row2 = threadIdx.x + i+d;
				col2 = threadIdx.y + j+e;
				idx2 = row2*N + col2;

			if (StrElement[(i+Yo/2)*Xo+j+Xo/2] == 1){
				//ElementResult = GPU_i[idx2];
				if  (min > shared_GPU_i[idx2])
					min = shared_GPU_i[idx2];
			}
		}
	}**/
	d = Yo/2;
	e = Xo/2;

	if ((row < d) || (row > (M - d-1))) goto End;
	
	int idx2, row2, col2;

	for (i=-d; i<=d; i++){
		for (j=-e; j<=e; j++){
			if (min == 0) continue;
				row2 = threadIdx.x + i+d;
				col2 = threadIdx.y + j+e;
				idx2 = row2*blockDim.y + col2;

			if (StrElement[(i+Yo/2)*Xo+j+Xo/2] == 1){
				//ElementResult = GPU_i[idx2];
				if  (min > shared_GPU_i[idx2])
					min = shared_GPU_i[idx2];
			}
		}
	}
	//if (blockIdx.x%2==0 && blockIdx.y%2==0)
	Erosiondata [idx] = min;
	
	End:;
}

__global__ void Dilation(uchar *GPU_i, uchar *Dilationdata, uchar *StrElement, int M, int N, int Xo, int Yo){

	extern __shared__ uchar shared_GPU_i[]; //does this get initialized at 0?  if not need to add 0 conditions to rest of conditionals
	
	int row = blockIdx.x * blockDim.x + threadIdx.x; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y; //column of image
	int idx = row*N + col; //which pixel in full 1D array
	int idy;
	int ids = (threadIdx.x+Yo/2)*blockDim.y+threadIdx.y+Xo/2;
	//uchar output = GPU_i[idx];
	//uchar ElementResult;

	int max;
	int i,j;
	int d = Yo/2;
	int e = Xo/2;
	max = 0;
	//printf("gridDim.x: %d \ngridDim.y: %d\n",gridDim.x,gridDim.y);
	if (row >= M || col >= N ) goto End;

	if ((blockIdx.x > 0) && (blockIdx.x < gridDim.x-1)){//if not first row or last row
		if ((blockIdx.y > 0) && (blockIdx.y < gridDim.y-1)){//if not first column or last column (general case)
			if ((threadIdx.x == 0) && (threadIdx.y == 0)){ //top left of block
				for (i=-d; i<=0; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == 0) && (threadIdx.y == (blockDim.y - 1))){ //top right
				for (i=-d; i<=0; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == 0)){ //bottom left
				for (i=0; i<=d; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == (blockDim.y - 1))){ //bottom right
				for (i=0; i<=d; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x < blockDim.x-1) && (threadIdx.y == 0)) { //left edge
				for (j=-e; j<=0; j++) {
					ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
					idy = (row)*N+col+j;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x < blockDim.x-1) && (threadIdx.y == blockDim.y-1)) { // Right edge
				for (j=0; j<=e; j++) {
					ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
					idy = (row)*N+col+j;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.y > 0) && (threadIdx.y < blockDim.y-1) && (threadIdx.x == 0)) { //top edge
				for (i=-d; i<=0; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.y > 0) && (threadIdx.y < blockDim.y-1) && (threadIdx.x == blockDim.x-1)) { //Bottom edge
				for (i=0; i<=d; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		if (blockIdx.y == 0){//leftmost column - left edge
			if ((threadIdx.x == 0) && (threadIdx.y == (blockDim.y - 1))){ //top right
				for (i=-d; i<=0; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == (blockDim.y - 1))){ //bottom right
				for (i=0; i<=d; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x < blockDim.x-1) && (threadIdx.y == blockDim.y-1)) { //right edge
				for (j=0; j<=e; j++) {
					ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j;
					idy = (row)*N+col+j;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.y < blockDim.x-1) && (threadIdx.x == 0)) { //top edge
				for (i=-d; i<=0; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.y < blockDim.x-1) && (threadIdx.x == blockDim.x-1)) { //Bottom edge
				for (i=0; i<=d; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		if (blockIdx.y == (gridDim.y - 1)){//if last column - right edge
			if ((threadIdx.x == 0) && (threadIdx.y == 0)){ //top left of block
				for (i=-d; i<=0; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == 0)){ //bottom left
				for (i=0; i<=d; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x < blockDim.x-1) && (threadIdx.y == 0)) {//left edge
				for (j=-e; j<=0; j++) {
					ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
					idy = (row)*N+col+j;
					shared_GPU_i[ids]= GPU_i[idy];
				}	
			}
			if ((threadIdx.y > 0) && (threadIdx.y <= blockDim.y-1) && (threadIdx.x == 0)) { //top edge
				for (i=-d; i<=0; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			if ((threadIdx.y > 0) && (threadIdx.y <= blockDim.y-1) && (threadIdx.x == blockDim.x-1)) { //Bottom edge
				for (i=0; i<=d; i++) {
					ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
	}
	
	if (blockIdx.x == 0){//if first row
		
		if ((blockIdx.y > 0) && (blockIdx.y < gridDim.y-1)){//if not first column or last column - top edge
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == 0)){ //bottom left
				for (i=0; i<=d; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == (blockDim.y - 1))){ //bottom right
				for (i=0; i<=d; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x < blockDim.x-1) && (threadIdx.y == 0)) {// Left edge

					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.x < blockDim.x-1) && (threadIdx.y == blockDim.y-1) ){ // Right edge
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.y > 0) && (threadIdx.y < blockDim.y-1) && (threadIdx.x == blockDim.x-1)) { //bottom edge

				for (i=0; i<=d; i++) {
					ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+e;
					idy = (row+i)*N+col;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
			else { //everything else
				ids = (threadIdx.x)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		if (blockIdx.y == 0){//if first column - top left corner

			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == (blockDim.y - 1))){ //bottom right
				for (i=0; i<=d; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+j;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x < blockDim.y-1) && (threadIdx.y == blockDim.y-1)) { //right edge
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x)*blockDim.y+threadIdx.y+j;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.y < blockDim.y-1) && (threadIdx.x == blockDim.x-1)) { //bottom edge
					for (i=0; i<=d; i++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y;
						idy = (row+i)*N+col;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			else { //everything else
				ids = (threadIdx.x)*blockDim.y+threadIdx.y;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
				
		if (blockIdx.y == (gridDim.y - 1)){//if last column - top right corner

			if ((threadIdx.x == (blockDim.x - 1)) && (threadIdx.y == 0)){ //bottom left
				for (i=0; i<=d; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			
			if ((threadIdx.x < blockDim.x-1) && (threadIdx.y == 0)) { // Left edge
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				
			}
			if ((threadIdx.y > 0) && (threadIdx.y <= blockDim.y-1) && (threadIdx.x == blockDim.x-1) ){ //Bottom edge

					for (i=0; i<=d; i++) {
						ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+e;
						idy = (row+i)*N+col;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			else { //everything else
				ids = (threadIdx.x)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
	}
	
	
	if (blockIdx.x == (gridDim.x - 1)){//if last row
		if ((blockIdx.y > 0) && (blockIdx.y < gridDim.y-1)){//if not first column or last column
			if ((threadIdx.x == 0) && (threadIdx.y == 0)){ //top left of block
				for (i=-d; i<=0; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x == 0) && (threadIdx.y == (blockDim.y - 1))){ //top right
				for (i=-d; i<=0; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			if ((threadIdx.x > 0) && (threadIdx.x <= blockDim.x-1) && (threadIdx.y == 0)) { // Left edge

					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.x > 0) && (threadIdx.x <= blockDim.x-1) && (threadIdx.y == blockDim.y-1) ){ // Right edge
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.y > 0) && (threadIdx.y < blockDim.y-1) && (threadIdx.x == 0)) { // Top edge
					for (i=-d; i<=0; i++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
						idy = (row+i)*N+col;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		
		if (blockIdx.y == 0){//if first column - bottom left corner
			if ((threadIdx.x == 0) && (threadIdx.y == (blockDim.y - 1))){ //top right
				for (i=-d; i<=0; i++) {
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}

			if ((threadIdx.x > 0) && (threadIdx.x <= blockDim.x-1) && (threadIdx.y == blockDim.y-1)) { // Right edge
					for (j=0; j<=e; j++) {
						ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.y < blockDim.x-1) && (threadIdx.x == 0)) { // Top edge
					for (i=-d; i<=0; i++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y;
						idy = (row+i)*N+col;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		if (blockIdx.y == (gridDim.y - 1)){//if last vertical block - bottom right corner
			if ((threadIdx.x == 0) && (threadIdx.y == 0)){ //top left of block
				for (i=-d; i<=0; i++) {
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row+i)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
				}
			}
			
			if ((threadIdx.x > 0) && (threadIdx.x <= blockDim.x-1) && (threadIdx.y == 0)) { // Left edge
					for (j=-e; j<=0; j++) {
						ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+j+e;
						idy = (row)*N+col+j;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			if ((threadIdx.y > 0) && (threadIdx.y <= blockDim.y-1) && (threadIdx.x == 0)) { // Top edge
					for (i=-d; i<=0; i++) {
						ids = (threadIdx.x+i+d)*blockDim.y+threadIdx.y+e;
						idy = (row+i)*N+col;
						shared_GPU_i[ids]= GPU_i[idy];
					}
			}
			else { //everything else
				ids = (threadIdx.x+d)*blockDim.y+threadIdx.y+e;
				idy = (row)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
	}
	
	/*if ((threadIdx.x<(blockDim.x-1)) && (threadIdx.y<(blockDim.y-1)))
		shared_GPU_i[ids]= GPU_i[idx];
	else {
		if (threadIdx.x==(blockDim.x-1) && (threadIdx.y<(blockDim.y-1))) {
			for (i=0; i<=d; i++) {
				ids = (threadIdx.x+i)*blockDim.y+threadIdx.y;
				idy = (row+i)*N+col;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		if (threadIdx.x<(blockDim.x-1) && (threadIdx.y==(blockDim.y-1))) {
			for (j=0; j<=e; j++) {
				ids = (threadIdx.x)*blockDim.y+threadIdx.y+j;
				idy = (row)*N+col+j+1;
				shared_GPU_i[ids]= GPU_i[idy];
			}
		}
		else {
			for (i=0; i<=d; i++) {
				for (j=0; j<=e; j++) {
					ids = (threadIdx.x+i)*blockDim.y+threadIdx.y+j;
					idy = (row+i)*N+col+j;
					shared_GPU_i[ids]= GPU_i[idy];
				}
			}
		}

	}**/
	//printf("ids : %d\n",ids);
	
	__syncthreads();

	
	/**if ((row < d) || (row > (M - d-1))) goto End;
	int idx2, row2, col2;

	for (i=-d; i<=d; i++){
		for (j=-e; j<=e; j++){
			if (min == 0) continue;
				row2 = threadIdx.x + i+d;
				col2 = threadIdx.y + j+e;
				idx2 = row2*N + col2;

			if (StrElement[(i+Yo/2)*Xo+j+Xo/2] == 1){
				//ElementResult = GPU_i[idx2];
				if  (min > shared_GPU_i[idx2])
					min = shared_GPU_i[idx2];
			}
		}
	}**/

	if ((row < d) || (row > (M - d-1))) goto End;
	int idx2, row2, col2;

	for (i=-d; i<=d; i++){
		for (j=-e; j<=e; j++){
			if (max == 255) continue;
				row2 = threadIdx.x + i+d;
				col2 = threadIdx.y + j+e;
				idx2 = row2*blockDim.y + col2;

			if (StrElement[(i+Yo/2)*Xo+j+Xo/2] == 1){
				//ElementResult = GPU_i[idx2];
				if  (max < shared_GPU_i[idx2])
					max = shared_GPU_i[idx2];
			}
		}
	}

	Dilationdata [idx] = max;
	
	End:;
}



__global__ void Threshold(uchar *GPU_i, uchar *GPU_o,  int M, int N, int Thresh)
{
    //long tn;            		     // My thread number (ID) is stored here
    //int row,col;
	unsigned char PIXVAL;
	double L,G;

    int rt = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int ct = blockIdx.y * blockDim.y + threadIdx.y;  // column of image
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


__global__ void Difference(uchar *Dilationdata, uchar *Erosiondata, uchar *GPU_o, int M, int N){
	int row = blockIdx.x * blockDim.x + threadIdx.x; //row of image
	int col = blockIdx.y * blockDim.y + threadIdx.y; //column of image
	int idx = row*N + col; //which pixel in full 1D array

	GPU_o [idx] = Dilationdata[idx] - Erosiondata[idx];

}			
	
int main(int argc, char *argv[]){
	float GPURuntimes[4]; //run times of the GPU code
	float ExecTotalTime, TfrCPUGPU, GPUTotalTime, TfrGPUCPU;
	cudaError_t cudaStatus;
	//; //output file name
	int i = 1;
	int *CPU_OutputArray = (int*) 0; //where the GPU should copy the output back to
	
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
		//Create CPU memory to store the output;
		//zero = Mat(M,N,CV_8UC1, Scalar(255)); //start by making every pixel white
		//sprintf(filename, "%s%d.png",argv[2],i);
		//imwrite(filename, zero);
		
		CPU_OutputArray = (int*) malloc(M*N*sizeof(int));
		if (CPU_OutputArray == NULL){
			fprintf(stderr, "Oops, cannot create CPU_OutputArray using malloc() ...\n");
			exit(EXIT_FAILURE);
		}
		//printf("malloc: %d\n",Xo*Yo*sizeof(uchar));
		StrElement = (uchar*) malloc(Xo*Yo*sizeof(uchar));
		memset(StrElement, 0, Xo*Yo*sizeof(uchar));

		CreateStrElement();

		//run it
		checkCuda(launch_helper(image, CPU_OutputArray, GPURuntimes),__LINE__);
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
		
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess){
			fprintf(stderr, "cudaDeviceReset failed!\n");
			free(CPU_OutputArray);
			exit(EXIT_FAILURE);
		}

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

		//printf("output: %s\n", output_filename.c_str());
		//output_filename[strl-5] = n;
		if (!imwrite(output_filename, result)) {
			fprintf(stderr, "couldn't write output to disk!\n");
			free(CPU_OutputArray);
			exit(EXIT_FAILURE);
		}
		printf("Saved image '%s', size = %dx%d (dims = %d).\n",
			   output_filename.c_str(), result.rows, result.cols, result.dims);
		free(CPU_OutputArray);
	//}
	exit(EXIT_SUCCESS);
		
}

cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes){
	
	cudaEvent_t time1, time2, time3, time4;
	int ucharGPUSize; // total size of 1 image in bytes
	int sharedMemSize;
	uchar *GPU_idata;
	uchar *GPU_odata;
	//uchar *GPU_zerodata;
	uchar *GPU_Dilationdata;
	uchar *GPU_Erosiondata;
	
	uchar *GPU_StrElement;

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
	/*cudaStatus = cudaMalloc((void**)&GPU_zerodata, ucharGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}*/
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

	cudaStatus = cudaMalloc((void**)&GPU_StrElement, Xo*Yo*sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	/*cudaStatus = cudaMemcpy(GPU_odata, zero.data, ucharGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyzero failed!\n");
		goto Error;
	}*/

	cudaStatus = cudaMemset(GPU_odata, 0, ucharGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(GPU_idata, image.data, ucharGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(GPU_StrElement, StrElement, Xo*Yo*sizeof(uchar), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaEventRecord(time2, 0);

	// Launch a kernel on the GPU with one thread for each pixel.
	threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
	numBlocks = dim3(ceil((float)M / threadsPerBlock.x), ceil((float)N / threadsPerBlock.y));
	//numBlocks = dim3(2, 2);
	sharedMemSize = (threadsPerBlock.x+Yo/2+1)*(threadsPerBlock.y+Xo/2+1)*sizeof(uchar);

	//EROSION AND DILATION
	/**
	Dilation<<<numBlocks, threadsPerBlock, sharedMemSize >>>(GPU_idata, GPU_Dilationdata, GPU_StrElement, M, N, Xo, Yo);

	// Check for errors immediately after kernel launch.
	checkCuda( cudaGetLastError(), __LINE__ );

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching addKernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}**/
	Erosion<<<numBlocks, threadsPerBlock, sharedMemSize >>>(GPU_idata, GPU_odata, GPU_StrElement, M, N, Xo, Yo);

	// Check for errors immediately after kernel launch.
	checkCuda( cudaGetLastError(), __LINE__ );

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	checkCuda( cudaDeviceSynchronize(), __LINE__);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching addKernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	// THEN TAKE THE DIFFERENCE
	/**
	Difference<<<numBlocks, threadsPerBlock, sharedMemSize >>>(GPU_Dilationdata, GPU_Erosiondata, GPU_odata, M, N);

	// Check for errors immediately after kernel launch.
	//checkCuda( cudaGetLastError(), __LINE__ );

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching addKernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	/*Threshold <<<numBlocks, threadsPerBlock >>>(GPU_idata, GPU_odata, M, N, Thresh);
	checkCuda( cudaGetLastError(), __LINE__ );
*/
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
	//cudaFree(GPU_zerodata);
	cudaFree(GPU_Dilationdata);
	cudaFree(GPU_Erosiondata);
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
	//ThreshLo++;
	return cudaStatus;
}
