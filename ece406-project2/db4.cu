/**refer to the following: https://en.wikipedia.org/wiki/Daubechies_wavelet
http://stackoverflow.com/questions/30773342/daubechies-orthogonal-wavelet-in-python
http://mlpy.sourceforge.net/docs/3.3/wavelet.html
http://wavelets.pybytes.com/wavelet/db4/
http://www.math-it.org/Publikationen/Wavelets.pdf
**/

//200 samples/sec
//Db4 makes N = 8
//phi (x) = sqrt(2) [h0*phi(2x) + h1*phi(2x-1) + h2*phi(2x-2) + h3*phi(2x-3) + h4*phi(2x-4) + h5*phi(2x-5) + h6*phi(2x-6) + h7*phi(2x-7) ]
//phi (0)
//phi (6) = sqrt(2) [h5*phi(7) + h6*phi(6) + h7*phi(5)]
//phi (7) = sqrt(2) * h7
//phi (1) = (1+sqrt(3))/2 --- phi(2) = (1-sqrt(3))/2 --- else if x is an int, phi(x) = 0
//psi (x) = sqrt(2) sum (-1)^k * h of 8-1-k * phi(2x - k) where k goes from 0 to 7
//looks like a QRS complex which spans 60-100ms
//inputs of x range from 0 to 7
//need to time scale it to handle t seconds over x so to handle 60ms-120ms(12 - 24 samples) , x step needs to be 0.25 - 0.125 (just take the 0.125 and be happy?)

float hcoeff(x){
	float array[8] = {0.2303778133, 0.7148465706, 0.6308807679, -0.0279837694, -0.1870348117, 0.0308413818,0.0328830117,-0.0105974018} ;
	if (x > length of array){
		return 0;
	}
	else {
		return array[x];
	}
}

float phi (x,N){
	
	float term;
	for i = 0; i <= 2*N - 1; i++{
		term = hcoeff(i) * phi(2x-i,N);
	}
	
	return SQRT2 * term;
}


#define SQRT2 1.41421356237309504880
#define SQRT3 1.73205080756887729352


/** actual coefficients TODO: scaling function 
#define H7 
#define H6 
#define H5 
#define H4 
#define H3 
#define H2 
#define H1 
#define H0 
**/

//LUT or calculate it out?
// x 	  = 0.0   0.5      1.0           1.5            2.0         2.5 3
// db2(x) = 0.0 -0.25 -0.36602540378 1.73205080757 -1.36602540378 -0.25 0
// not db4(x) as assumed

__device__ __host__ float phi(float x){//WARNING: recursive nature see last else clause
    //test for 1 2 or integer
    int upperx = ceilf(x);
    int lowerx = floorf(x);
    float result;
    if ((upperx - x) < 0.00000001){ // 8 digits accuracy is single precision
		//is close enough to be considered an integer   
		switch (upperx){
			case 1 : 
				result = (1 + SQRT3)*0.5; //to be recalculated
				break;
			case 2 : 
				result = (1 - SQRT3)*0.5; //to be recalculated
				break;
			case 3 : 
				break;
			case 4 : 
				break;
			case 5 : 
				break;
			case 6 : 
				break;
			case 7 : 
				break;
			default :
				result = 0;
				break;
		}
	}
	else if ((x - lowerx) < 0.00000001){ // 8 digits accuracy is single precision
		//is close enough to be considered an integer   
		switch (lowerx){
			case 1 : 
				result = (1 + SQRT3)*0.5; //to be recalculated
				break;
			case 2 : 
				result = (1 - SQRT3)*0.5; //to be recalculated
				break;
			case 3 : 
				break;
			case 4 : 
				break;
			case 5 : 
				break;
			case 6 : 
				break;
			case 7 : 
				break;
			default :
				result = 0;
				break;
		}
	}
    else{//fix this ASAP probably not even executable
	result = SQRT2 * (H0*phi(2*x) + H1*phi(2x-1) + H2*phi(2x-2) + H3*phi(2x-3));
    }
    return result;
}

__device__ __host__ float db4_point(float x){//evaluates db4 at a specific point
    float term1 = H7 * phi(2 * x - 0);
    float term2 = H6 * phi(2 * x - 1);
    float term3 = H5 * phi(2 * x - 2);
    float term4 = H4 * phi(2 * x - 3);
	float term5 = H3 * phi(2 * x - 4);
	float term6 = H2 * phi(2 * x - 5);
	float term7 = H1 * phi(2 * x - 6);
	float term8 = H0 * phi(2 * x - 7);
	//should have 8 of these terms
    float term9 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8;
    return SQRT2 * term9;
}

__global__ void db4_wavelet(float* out_signal, float start, float step){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float i = start + ((float) tid) * step;
  out_signal[tid] = db4_point(i);
}

__device__ __host__ void point_cross_correlate(float * result_point, float * signal, float * wavelet_signal, int point, int signal_size, int points_per_wavelet){
  float result = 0.0;
  for (int i = -points_per_wavelet / 2; i < points_per_wavelet / 2; i++) {
    if (point + i >= signal_size) {
      break;
    } else if (point + i < 0) {
      continue;
    }
    result += signal[point + i] * hat_signal[i + points_per_wavelet / 2];
  }
  * result_point = result;
}

__global__ void wavelet_cross_correlate(float* out_signal, float * in_signal, float* wavelet_signal, int signal_size, int points_per_wavelet){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  point_cross_correlate(&out_signal[idx], in_signal, wavelet_signal, idx, signal_size, points_per_wavelet);
}
