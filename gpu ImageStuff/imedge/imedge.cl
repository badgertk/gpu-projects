#define EDGE			255
#define NOEDGE			0
#define PI				3.14159265359

__kernel void GaussianFilter(__global uchar* GPU_i, __global double* Gauss_i, int M, int N){
	int idx = get_global_id(0);
	int col = (idx) % N; //column of image
	int row = (idx - col)/N; //row of image
	int i,j;
	double G;
	double Gauss[5][5] = {	{ 2, 4,  5,  4,  2 },
						{ 4, 9,  12, 9,  4 },
						{ 5, 12, 15, 12, 5 },
						{ 4, 9,  12, 9,  4 },
						{ 2, 4,  5,  4,  2 }	};
	
	if ((row < 2) || (row > (M - 3))) return;
	//col = 2;
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
	
	Gauss_i[idx] = G/ (double)159.00;	
	//Gauss_i[idx] = GPU_i[idx];
	return;

}

__kernel void Sobel(__global double* Gradient, __global double* Theta, __global double* Gauss, int M, int N){
	int idx = get_global_id(0);
	int col = (idx) % N; //column of image
	int row = (idx - col)/N; //row of image
	int i,j;
	double GX,GY;
	//printf("row = %d, col = %d", row, col);
	
double Gx[3][3] = {		{ -1, 0, 1 },
						{ -2, 0, 2 },
						{ -1, 0, 1 }	};

double Gy[3][3] = {		{ -1, -2, -1 },
						{  0,  0,  0 },
						{  1,  2,  1 }	};
	
	if ((row<1) || (row>(M-2))) return;
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
				GX = GX + Gauss[idx2] * Gx[i+1][j+1];
				//printf("Gauss[] = %f", Gauss[idx2]); //a lot of 124.92
				GY = GY + Gauss[idx2] * Gy[i+1][j+1];
				//printf("Gy[] = %f", Gy[i+1][j+1]);
			}
		}
		
		Gradient[idx] = sqrt(GX*GX+GY*GY);
		//printf("GX = %f GY = %f Gradient = %f", GX, GY, sqrt(GX*GX+GY*GY)); //GX always = 0 and GY always = 499?
		Theta[idx] = atan(GX/GY) * 180.0/PI;
	}
	//Gradient[idx] = Gauss[idx];
	return;
}

__kernel void Threshold(__global uchar* Theta, __global double* Gradient, __global uchar* GPU_o, int M, int N){
	int idx = get_global_id(0);
	int col = (idx) % N; //column of image
	int row = (idx - col)/N; //row of image
	uchar PIXVAL;
	double L,H,G,T;
	
	int ThreshLo = 8; int ThreshHi = 15;
	
	/*
	int row46, col46, idx46; //left right
	int row28, col28, idx28; //top bottom
	int row19, col19, idx19; //lower left upper right
	int row37, col37, idx37; //lower right upper left
	*/
	
	if ((row<1) || (row>(M-2))) return;
	//col = 1;
	L = (double) ThreshLo; H = (double)ThreshHi;
	G = Gradient[idx];
	PIXVAL = NOEDGE;
	if (G <= L){
		PIXVAL = NOEDGE;
	} else if (G >= H){
		//printf("G = %f and H = %f", G, H);
		PIXVAL = EDGE;
	} else{
			//printf("GOT IN HERE?");
		T = Theta [idx];
		if ((T < -67.5) || (T > 67.5)){
			//look left and right
			PIXVAL = ((Gradient[row*N + col - 1] > H) || (Gradient[row*N + col + 1] > H)) ? EDGE:NOEDGE;
		} else if ((T >= -22.5) && (T <= 22.5)){
			//look top and bottom
			PIXVAL = ((Gradient[(row - 1)*N + col] > H) || (Gradient[(row + 1)*N + col] > H)) ? EDGE:NOEDGE;
		} else if ((T > 22.5) && (T <= 67.5)){
			//look upper right and lower left
			PIXVAL = ((Gradient[(row - 1)*N + col + 1] > H) || (Gradient[(row + 1)*N + col - 1] > H)) ? EDGE:NOEDGE;
		} else if ((T >= -67.5) && (T < -22.5)){
			//look upper left and lower right
			PIXVAL = ((Gradient[(row - 1)*N + col - 1] > H) || (Gradient[(row + 1)*N + col + 1] > H)) ? EDGE:NOEDGE;
		}
	}

	GPU_o[idx] = PIXVAL;
	return;
}
