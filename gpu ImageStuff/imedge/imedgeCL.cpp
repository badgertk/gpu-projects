#include <CL/cl.h>
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

#define MEM_SIZE 128
#define MAX_SOURCE_SIZE 0x100000
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define DATA_SIZE 1024

// OpenCV stuff (note: C++ not C):
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

// Courtesy of Chase
void getWorkGroupSizes(cl_device_id device_id, cl_kernel kernel, size_t * local, size_t * global, size_t desired_global) {
  int err;
  size_t max_work_group_size;
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
  // Get the maximum work group size for executing the kernel on the device
  //
  err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), local, NULL);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to retrieve kernel work group info! %d\n", err);
      exit(1);
  }
  * global = desired_global + (desired_global % max_work_group_size);
}

void listDevices() {
  int i, j;
  char* value;
  size_t valueSize;
  cl_uint deviceCount;
  cl_device_id* devices;
  cl_uint maxComputeUnits;

  // get all devices
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
  devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
  assert(devices);
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

  //print attributes of each device
  printf("Available devices : \n");
  for (j = 0; j < deviceCount; j++) {
      // print device name
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
      value = (char*) malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
      printf("[%d]: %s\n", j, value);
      free(value);
  }
  printf("----------------------------\n");

  free(devices);
}

void listPlatforms(){
    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id *platforms;
    const char* attributeNames[5] = { "Name", "Vendor", "Version", "Profile", "Extensions"};
    const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
    const int attributeCount = sizeof(attributeNames) / sizeof(char*);
 
    // get platform count
    clGetPlatformIDs(5, NULL, &platformCount);
 
    // get all platforms
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);
 
    // for each platform print all attributes
    for (i = 0; i < platformCount; i++) {
        printf("\n %d. Platform \n", i+1);
 
        for (j = 0; j < attributeCount; j++) {
 
            // get platform attribute value size
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
 
            // get platform attribute value
            clGetPlatformInfo(platforms[i], attributeTypes[j], valueSize, value, NULL);
 
            printf("  %d.%d %s: %s\n", i+1, j+1, attributeNames[j], value);
            free(value);
        }
        printf("\n");
    }
    free(platforms);
}

void printDeviceInfo(cl_device_id device) {
  char* value;
  size_t valueSize;
  cl_uint platformCount;
  cl_uint maxComputeUnits;


  // print device name
  clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
  value = (char*) malloc(valueSize);
  clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
  printf("\nDevice: %s\n", value);
  free(value);

  // print hardware device version
  clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &valueSize);
  value = (char*) malloc(valueSize);
  clGetDeviceInfo(device, CL_DEVICE_VERSION, valueSize, value, NULL);
  printf("  Hardware version: %s\n", value);
  free(value);

  // print software driver version
  clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &valueSize);
  value = (char*) malloc(valueSize);
  clGetDeviceInfo(device, CL_DRIVER_VERSION, valueSize, value, NULL);
  printf("  Software version: %s\n", value);
  free(value);

  // print c version supported by compiler for device
  clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
  value = (char*) malloc(valueSize);
  clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
  printf("  OpenCL C version: %s\n", value);
  free(value);

  // print parallel compute units
  clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
          sizeof(maxComputeUnits), &maxComputeUnits, NULL);
  printf("  Parallel compute units: %d\n\n", maxComputeUnits);
}

inline void checkErr(cl_int err, const char * name){
	if (err != CL_SUCCESS){
		printf("ERROR: %c ( %d )\n", name, err);
		exit(EXIT_FAILURE);
	}
}


int main(int argc, char *argv[]){
	int err; //error code
	size_t global;
	size_t local;

	cl_device_id device_id;
	cl_context context;
	cl_command_queue command_queue;
	cl_mem input; // source image
	cl_mem output; // output image
	cl_mem gradient;
	cl_mem theta;
	cl_mem gauss;
	cl_program program;
	cl_kernel kernel1, kernel2, kernel3;
	cl_event event1, event2, event3;
	
    struct timeval 		t;
    double         		HtoD_start, HtoD_end, DtoH_start, DtoH_end;
	double				HtoD_time, DtoH_time;
    char* value;
    size_t valueSize;
    const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
	float TotalExecTime;
	
	int i = 0;
	printf("========== PRINTOUT ===========\n");
	// Parse input args:
	if ( argc != 4 ) {
		printf("INCORRECT USAGE!\n");
		printf("Usage: %s <input image file name> <output image file name> <device number>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	int device = atoi(argv[3]);

	FILE *fp;
	char fileName[] = "./imedge.cl"; // .cl file name here
	char *source_str;
	size_t source_size;

	// Connect to a compute device

	cl_uint num_platforms;
    // Get platform count
    clGetPlatformIDs(5, NULL, &num_platforms); // Maximum of platforms that will be looked for = 5
 
    // get all platforms
    cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);
	printf("Number of Platforms = %d\n", num_platforms);	
	listPlatforms();
    cl_platform_id * platform_id = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL); 
	if (err != CL_SUCCESS) {
		printf("Error: Failed to select a platform!\n");
		return EXIT_FAILURE;
	}
    else{
	
		clGetPlatformInfo(platforms[0], attributeTypes[2], 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
 
        // Get platform attribute value
        clGetPlatformInfo(platforms[0], attributeTypes[2], valueSize, value, NULL);
		printf("Successfully selected platform %s\n", value);
        free(value);
	}

	// Now for devices
    cl_uint num_devices;
    clGetDeviceIDs(* platforms, CL_DEVICE_TYPE_ALL, 0, NULL, & num_devices);
    cl_device_id * devices = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
          
	if (! devices) { 
		printf("Number of Devices = %d\n", num_devices);
		printf("Error: Failed to create a device group due to no devices found!\n");
		return EXIT_FAILURE;
	}
    err = clGetDeviceIDs(* platforms, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to create a device group!\n");
		printf("Error: Could not get deviceID!\nValid options are:\n");
		listDevices();
		return EXIT_FAILURE; 
    }
    if (device < num_devices){
		device_id = devices[device];
		// print device name
		printDeviceInfo(device_id);
		value = (char*) malloc(valueSize);
		clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, value, NULL);
		printf("Successfully selected device %s\n", value);
		free(value);
	}
    else {
		printf("Error: Device requested does not exist! Valid options are:\n");
		listDevices();
		return EXIT_FAILURE;      
    }

	int a = 0;
	// Load input image:
	Mat image;	// see http://docs.opencv.org/modules/core/doc/basic_structures.html#mat
	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	int M = image.rows;
	int N = image.cols;
	printf("M = %d N = %d\n", M , N);
	size_t TotalSize = M * N * sizeof(unsigned char); //3200*2400 greyscale not RGB
	size_t doubleSize = M * N * sizeof(double);
    if(! image.data ){
      fprintf(stderr, "Could not open or find the image.\n");
      exit(EXIT_FAILURE);
    }
    //image.convertTo(image, CV_32FC1);
    printf("\nLoaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);

	// Load the source code containing the kernel
	fp = fopen(fileName, "r");
	if (!fp){
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	
    unsigned char *CPU_InputArray = (unsigned char *)malloc(TotalSize);
    unsigned char *CPU_OutputArray = (unsigned char *)malloc(TotalSize);

	if ((CPU_InputArray == NULL) || (CPU_OutputArray == NULL)) {
		fprintf(stderr, "Could not create I/O array(s) using malloc() ...\n");
		exit(EXIT_FAILURE);
    }
	fclose(fp);

	gettimeofday(&t, NULL);
    HtoD_start = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	
	memcpy(CPU_InputArray, image.data, TotalSize);
	
	gettimeofday(&t, NULL);
    HtoD_end = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	
	HtoD_time +=(HtoD_end - HtoD_start)/1000.00;
	// Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context){
        printf("Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
	
    // Create a command queue
    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!command_queue)
    {
        printf("Failed to create a command queue!\n");
        return EXIT_FAILURE;
    }

	// Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &err);
    if (!program){
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    // Build the program executable
		printf("got down here\n\n\n");
    err = clBuildProgram(program, 0, NULL , NULL, NULL, NULL);
	//checkErr(err)
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char *log = (char *) malloc(len);
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, log, NULL);
        printf("%s\n", log);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    kernel1 = clCreateKernel(program, "GaussianFilter", &err);
    kernel2 = clCreateKernel(program, "Sobel", &err);
    kernel3 = clCreateKernel(program, "Threshold", &err);
    //kernel = clCreateKernel(program, "nothing", &err);
    
	if (!kernel1 || !kernel2 || !kernel3 || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  doubleSize, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TotalSize, NULL, NULL);
    gradient = clCreateBuffer(context, CL_MEM_READ_WRITE, doubleSize, NULL, NULL);
    theta = clCreateBuffer(context, CL_MEM_READ_WRITE, doubleSize, NULL, NULL);
    gauss = clCreateBuffer(context, CL_MEM_READ_WRITE, doubleSize, NULL, NULL);
    if (!input || !output){
        printf("Failed to allocate device buffers!\n");
        exit(1);
    }    
    
    // Write our data set into the input array in device memory 
    err = clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, TotalSize, CPU_InputArray , 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Failed to write to source buffer!\n");
        exit(1);
    }
	
	// Set the arguments to our compute kernel FIXED?
    err = 0;
    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &gauss);
    err |= clSetKernelArg(kernel1, 2, sizeof(int), &M);
    err |= clSetKernelArg(kernel1, 3, sizeof(int), &N);
    if (err != CL_SUCCESS){
        printf("Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
	getWorkGroupSizes(device_id, kernel1, &local, &global, M * N);

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device

	// Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel1, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    err = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL, &global, &local, 0, NULL, &event1);
    if (err){
        printf("Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
	// Wait for everything in the command queue to be executed before reading back results
    clFinish(command_queue);
	clWaitForEvents(1 , &event1);

	// Set the arguments to our compute kernel FIXED?
    err = 0;
    err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &gradient);
    err |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &theta);
    err |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &gauss);
    err |= clSetKernelArg(kernel2, 3, sizeof(int), &M);
    err |= clSetKernelArg(kernel2, 4, sizeof(int), &N);
    if (err != CL_SUCCESS){
        printf("Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
	getWorkGroupSizes(device_id, kernel2, &local, &global, M * N);

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device

	// Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel2, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    err = clEnqueueNDRangeKernel(command_queue, kernel2, 1, NULL, &global, &local, 0, NULL, &event2);
    if (err){
        printf("Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
	// Wait for everything in the command queue to be executed before reading back results
    clFinish(command_queue);
	clWaitForEvents(1 , &event2);
	
	// Set the arguments to our compute kernel FIXED?
    err = 0;
    err = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &theta);
    err |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &gradient);
    err |= clSetKernelArg(kernel3, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel3, 3, sizeof(int), &M);
    err |= clSetKernelArg(kernel3, 4, sizeof(int), &N);
    if (err != CL_SUCCESS){
        printf("Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
	getWorkGroupSizes(device_id, kernel3, &local, &global, M * N);

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device

	// Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel3, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    err = clEnqueueNDRangeKernel(command_queue, kernel3, 1, NULL, &global, &local, 0, NULL, &event3);
    if (err){
        printf("Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
	// Wait for everything in the command queue to be executed before reading back results
    clFinish(command_queue);
	clWaitForEvents(1 , &event3);
	
	gettimeofday(&t, NULL);
    DtoH_start = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	
    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer(command_queue, output , CL_TRUE, 0, TotalSize, CPU_OutputArray, 0, NULL, NULL);  
    if (err != CL_SUCCESS){
        printf("Failed to read output array! %d\n", err);
        exit(1);
    }
	gettimeofday(&t, NULL);
    DtoH_end = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	// Display the output image:
	Mat result = Mat(M, N, CV_8UC1, CPU_OutputArray);
	//Mat zero = Mat(M,N,CV_8UC1, Scalar(255));
	// and save it to disk:
	char output_filename[100];
	sprintf(output_filename,"%s", argv[2]);
	imwrite(output_filename,result);
	if (!imwrite(output_filename, result)){
		fprintf(stderr, "Couldn't write output to disk!\n");
		exit(EXIT_FAILURE);
	}

	
	DtoH_time += (DtoH_end - DtoH_start)/1000.00;
	printf("\nSaved image '%s', size = %dx%d (dims = %d).\n", output_filename, result.rows, result.cols, result.dims);

	cl_ulong time_start, time_end;
    double GPUTime;

    clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    GPUTime += (time_end - time_start)/1000000.0;
	TotalExecTime = HtoD_time + GPUTime + DtoH_time;
	printf("Host to Device Transfer Time : %0.3f ms\n", HtoD_time);
	printf("GPU Execution Time : %0.3f ms\n", GPUTime);
	printf("Device to Host Transfer Time : %0.3f ms\n", DtoH_time);
	free(CPU_OutputArray);
	free(CPU_InputArray);

	printf("Total Execution Time : %0.3f ms\n", TotalExecTime);
    printf("========= EXECUTION COMPLETE =============\n");
    
    // Shutdown and cleanup

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    clReleaseKernel(kernel3);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
	free(source_str);
 
	return 0;
}
