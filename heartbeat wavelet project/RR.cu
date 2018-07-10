#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
// makefile associated with this file
// salloc -t 5 -A ece406 -p ece406 --gres=gpu:1 srun ./RR 

//CUDA STUFF:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define TESTNUM			320000
#define SAMPLETOMIN		0.3
#define SAMPLINGRATE	200
#define BOXSIZE			32
//5 decimal points
#define DECACC			100000
#define ONEOVERDECACC 	0.00001
#define THREADSPERBLOCK BOXSIZE * BOXSIZE
#define MAXTHREADS		64
FILE *fs;  //declare the file pointer file source (ie: 9004RR.csv)
FILE *ft;  // result csv file for QTClock generation

int count;

cudaError_t launch_helper(int* featureid, int* sample, int* interval, int* hour, int* minute, float* second, float* Runtimes, int mode);
__device__ int int_3_median_index( int * in_signal, int index);
__device__ int int_3_median_value(int * in_signal, int index);
__global__ void int_3_median_filter( int * out_signal, int * in_signal);

//kernels
__global__ void DuplicateShift(int * featureid_in, int * featureid_out, int * sampleid_in, int * sampleid_out){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	featureid_out[tid] = featureid_in[tid+1];
	sampleid_out[tid] = sampleid_in[tid+1];
}

__global__ void RtoR(int * GPU_sampledata1, int * GPU_sampledata2, int * GPU_HR){
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	GPU_HR[tid] = int((GPU_sampledata2[tid] - GPU_sampledata1[tid])*SAMPLETOMIN);

	//End:;
}


__global__ void QT(int * GPU_featureid1, int * GPU_featureid2, int * GPU_sampledata1, int * GPU_sampledata2, int * GPU_QT){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int diff = (GPU_featureid2[tid] - GPU_featureid1[tid]);
	GPU_QT[tid] = (diff == 5) ? (GPU_sampledata2[tid] - GPU_sampledata1[tid]):(GPU_sampledata2[tid+1] - GPU_sampledata1[tid+1]);
}

__global__ void SampletoHMS(int * sampledata, int * hour, int * minute, float * second){
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int i;

	hour[tid] = int(sampledata[tid] * 0.00000138888);
	minute[tid] = int(sampledata[tid] * 0.00008333333) % (60);
	i = int(sampledata[tid] * 0.005 * DECACC) % (60* DECACC);
	second[tid] = i * ONEOVERDECACC;
	//second[tid] = fmod(float(sampledata[tid] * 0.005), float(60));
	
}

int main(int argc, char *argv[]){
	float GPURuntimes[4] = {0,0,0,0}; //run times of the GPU code
	float ExecTotalTime=0, TfrCPUGPU=0, GPUTotalTime=0, TfrGPUCPU=0;
	cudaError_t cudaStatus;
	clock_t begin, end;
	double time_spent;

	begin = clock();

	if (argc != 1){
		printf("Improper Usage!\n");
		printf("Usage: %s \n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	char buffer[100];
	long id;
	int lead;
	char feature[3];
	int featureid[TESTNUM], hour[TESTNUM], minute[TESTNUM];
	int sample[TESTNUM], HR[TESTNUM];
	float second[TESTNUM];
	// list of csv files to read
	const char *fsCSVQueue[] = {"lead1RR.csv","lead2RR.csv","lead3RR.csv","lead1QT.csv","lead2QT.csv","lead3QT.csv","lead1QRS.csv","lead2QRS.csv","lead3QRS.csv","lead1ST.csv","lead2ST.csv","lead3ST.csv",};
	const char *ftCSVQueue[] = {"lead1GPUHR.csv","lead2GPUHR.csv","lead3GPUHR.csv","lead1GPUQT.csv","lead2GPUQT.csv","lead3GPUQT.csv","lead1GPUQRS.csv","lead2GPUQRS.csv","lead3GPUQRS.csv","lead1GPUST.csv","lead2GPUST.csv","lead3GPUST.csv"};
	int i = 0;
	printf("-----------------------------------------------------------------\n");
	printf("Stages 0-2 are RR, Stages 3-5 are QT, Stages 6-8 are QRS, Stages 9-11 are ST.\n");
	printf("-----------------------------------------------------------------\n");

	for (i = 0; i < 12; i++){
		int j = 0;
		count = 0;
		fs = fopen(fsCSVQueue[i], "r");
		//--------------------------------------------------------------------------------------------------------
		if (fs == NULL){
			printf("Cannot open file fs!\n");
			return 0;
		}
		
		fgets(buffer, 100, fs); //ignore first line
		//should be "id,lead,sample,feature"
		//printf("%s\n", buffer);
		
		while ((fscanf(fs, "%ld,%d,%llu,%s", &id, &lead, &sample[j], &feature)) != EOF){
			//convert feature deliniations to featureids
			if (feature[0] == 'R'){featureid[j] = 2;}
			else if (feature[0] == 'N'){featureid[j] = 1;}
			else if (feature[0] == 't' & feature[1] == ')'){featureid[j] = 6;}
			else if (feature[0] == '(' & feature[1] == 't'){featureid[j] = 11;}
			else if (feature[0] == ')'){featureid[j] = 6;}
			j++;
		}
		
		count = j;
		printf("count = %d\n", j);
		fclose(fs);
		//run it
		//0 = Heartrate
		//1 = QT interval
		//2 = QRS interval
		//3 = ST interval
		int mode = i / 3; 
		cudaStatus = launch_helper(featureid, sample, HR, hour, minute, second, GPURuntimes, mode);
		cudaStatus = cudaSuccess;
		if (cudaStatus != cudaSuccess){
			fprintf(stderr, "launch_helper failed!\n");
			exit(EXIT_FAILURE);
		}

		//printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \nSum of Iteration = %5.2f ms\n", GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
		ExecTotalTime += GPURuntimes[0];
		TfrCPUGPU += GPURuntimes[1];
		GPUTotalTime += GPURuntimes[2];
		TfrGPUCPU += GPURuntimes[3];

		printf("Stage %d Execution Time = (%5.2f ms, %5.2f ms, %5.2f ms)\n", i, GPURuntimes[1], GPURuntimes[2], GPURuntimes[3]);
		
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess){
			fprintf(stderr, "cudaDeviceReset failed!\n");
			//free(CPU_OutputArray);
			exit(EXIT_FAILURE);
		}
		//--------------------------------------------------------------------------------------------------------

		ft = fopen(ftCSVQueue[i], "w");
			
		if (ft == NULL){
			printf("Cannot open file ft!\n");
			return 0;
		}
		fprintf(ft, "time, interval\n");

		for (j = 0; j < count - 1; j++){
			//fprintf(ft, "%llu, %llu\n", sample[j], sample2[j]);
			fprintf(ft, "%d:%d:%2.20f ,%llu\n", hour[j], minute[j], second[j], HR[j]);
		}
		fclose(ft);
		
	}
	// FIX THIS LAST
	
	printf("-----------------------------------------------------------------\n");
	printf("\nTfr CPU -> GPU Time = %5.2f ms\n", TfrCPUGPU);
	printf("GPU Execution Time = %5.2f ms \n", GPUTotalTime);
	printf("Tfr GPU -> CPU Time = %5.2f ms\n", TfrGPUCPU);
	printf("Total GPU Time = %5.2f ms\n", ExecTotalTime);
	printf("-----------------------------------------------------------------\n");
	printf("reached end of program\n");
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("TOTAL TIME SPENT = %f seconds\n", time_spent);
	return 0;
}

cudaError_t launch_helper(int* featureid, int* sample, int* interval, int* hour, int* minute, float* second, float* Runtimes, int mode){
		
	cudaEvent_t time1, time2, time3, time4;
	cudaStream_t stream[2];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);
	dim3 threadsPerBlock;
	dim3 numBlocks;
	int intGPUSize,floatGPUSize;
	int *GPU_featureiddata1, *GPU_featureiddata2, *GPU_Hour, *GPU_Minute;
	float *GPU_Second;
	int *GPU_sampledata1, *GPU_sampledata2, *GPU_RtoR, *GPU_Interval;
	
	threadsPerBlock = dim3(BOXSIZE,BOXSIZE);
	numBlocks = dim3(count/threadsPerBlock.x,1);
	//printf("numBlocks = (%d, %d)\n", count/threadsPerBlock.x,1);
	//printf("threadsPerBlock = (%d, %d)\n", BOXSIZE,BOXSIZE);
	
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
	intGPUSize = count * sizeof(int);
	floatGPUSize = count * sizeof(float);
	
	cudaStatus = cudaMalloc((void**)&GPU_featureiddata1, intGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_featureiddata2, intGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_sampledata1, intGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_sampledata2, intGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_RtoR, intGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_Interval, intGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_Hour, intGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_Minute, intGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&GPU_Second, floatGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(GPU_featureiddata1, featureid, intGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPU_sampledata1, sample, intGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	
	
	//launch some kernels
	cudaEventRecord(time2, 0);
	DuplicateShift<<<numBlocks,threadsPerBlock>>>(GPU_featureiddata1, GPU_featureiddata2, GPU_sampledata1, GPU_sampledata2);

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

	if (mode == 0){ //RR take differences from arrays of the same featureid

		RtoR<<<numBlocks,threadsPerBlock, 0, stream[0]>>>(GPU_sampledata1, GPU_sampledata2, GPU_RtoR);
		SampletoHMS<<<numBlocks,threadsPerBlock, 0 , stream[1]>>>(GPU_sampledata2, GPU_Hour, GPU_Minute, GPU_Second);
		int_3_median_filter<<<numBlocks,threadsPerBlock, 0, stream[1]>>>(GPU_RtoR, GPU_RtoR);
	
	}
	
	if (mode != 0){ //everything else, take difference between two different featureids
		QT<<<numBlocks,threadsPerBlock, 0, stream[0]>>>(GPU_featureiddata1, GPU_featureiddata2, GPU_sampledata1, GPU_sampledata2, GPU_Interval);
		SampletoHMS<<<numBlocks,threadsPerBlock, 0 , stream[1]>>>(GPU_sampledata2, GPU_Hour, GPU_Minute, GPU_Second);
		int_3_median_filter<<<numBlocks,threadsPerBlock, 0, stream[1]>>>(GPU_Interval, GPU_Interval);
		
	}
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
	if (mode == 0){
		cudaStatus = cudaMemcpy(interval, GPU_RtoR, intGPUSize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			goto Error;
		}
	}
	if (mode != 0){
		cudaStatus = cudaMemcpy(interval, GPU_Interval, intGPUSize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			goto Error;
		}
	}

	cudaStatus = cudaMemcpy(hour, GPU_Hour, intGPUSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(minute, GPU_Minute, intGPUSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(second, GPU_Second, floatGPUSize, cudaMemcpyDeviceToHost);
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

	cudaFree(GPU_featureiddata1);
	cudaFree(GPU_featureiddata2);
	cudaFree(GPU_sampledata1);
	cudaFree(GPU_sampledata2);
	cudaFree(GPU_RtoR);
	cudaFree(GPU_Interval);
	cudaFree(GPU_Hour);
	cudaFree(GPU_Minute);
	cudaFree(GPU_Second);
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
	return cudaStatus;
}

// code from project 1 
__device__ int
int_3_median_index(
  int * in_signal,
  int index)
{
  // Cardinality (3 is largest) -> (compare1, compare2, compare3) [position]
  // 1 2 3 -> 1 1 1 [1] 01
  // 1 3 2 -> 1 1 0 [2] 10
  // 2 1 3 -> 0 1 1 [0] 00
  // 2 3 1 -> 1 0 0 [0] 00
  // 3 1 2 -> 0 0 1 [2] 10
  // 3 2 1 -> 0 0 0 [1] 01
  int lookup[8] = { 1, 2, 0, 0, 0, 0, 2, 1};
  int compare1 = in_signal[index] < in_signal[index + 1];
  int compare2 = in_signal[index] < in_signal[index + 2];
  int compare3 = in_signal[index + 1] < in_signal[index + 2];
  int compare_mask = (compare1 << 2) | (compare2 << 1) | compare3;
  return lookup[compare_mask];
}

__device__ int
int_3_median_value(
  int * in_signal,
  int index
  )
{
  return in_signal[index + int_3_median_index(in_signal, index)];
}

__global__ void
int_3_median_filter(
  int * out_signal,
  int * in_signal)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  out_signal[tid] = int_3_median_value(in_signal, tid);
}
