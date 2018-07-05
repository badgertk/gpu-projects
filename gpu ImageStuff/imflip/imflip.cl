__kernel void FlipImageH(__global uchar* GPU_i, __global uchar* GPU_o, int M, int N){
	int idx = get_global_id(0);
if (idx >= M){// do nothing if I'm outside of the image
		return;
	}  
	/**int start = idx*N;        // first byte of row in 1D array
	int end = start + N - 1;  // last byte of row in 1D array

	int col;
	for (col=0; col<N; col++) {
		GPU_o[start+col] = GPU_i[end-col];
	}**/
 
  int start = idx*N*3;        // first byte of row in 1D array
  int end = start + N*3 - 1;  // last byte of row in 1D array

  int col;
  for (col=0; col<N*3; col+=3) {
    GPU_o[start+col]   = GPU_i[end-col-2];
    GPU_o[start+col+1] = GPU_i[end-col-1];
    GPU_o[start+col+2] = GPU_i[end-col];
  }
}

__kernel void nothing(__global uchar* GPU_i, __global uchar* GPU_o, int M, int N){
	int idx = get_global_id(0);
	GPU_o[idx] = GPU_i[idx];
}