__kernel void rotate_kernel(__global uchar* GPU_i, __global uchar* GPU_o, int M, int N, int i, int j){

	int idx = get_global_id(0);
	int col = idx % N; //column of image
	int row = (idx - col)/N; //row of image
	//int idx = row*N + col; //which pixel in full 1D array

	uchar output = GPU_i[idx];


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
	GPU_o[idx2] = output;

}
__kernel void nothing(__global uchar* GPU_i, __global uchar* GPU_o, int M, int N, int i, int j){
	int idx = get_global_id(0);
	GPU_o[idx] = GPU_i[idx];
}
