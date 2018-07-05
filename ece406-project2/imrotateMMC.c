#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "ImageStuff.h"

#define MAXTHREADS   128

long  			NumThreads;         		// Total number of threads working in parallel
int 	     	ThParam[MAXTHREADS];		// Thread parameters ...
double			RotAngle;					// rotation angle
pthread_t      	ThHandle[MAXTHREADS];		// Thread handles
pthread_attr_t 	ThAttr;						// Pthread attrributes
pthread_mutex_t	image_lock;					// Pthread mutex lock
pthread_mutex_t nextrow_lock;				// Pthread mutex lock
void* (*RotateFunc)(void *arg);				// Function pointer to rotate the image (multi-threaded)
int NextRowtoProcess = 0;

unsigned char**	TheImage;					// This is the main image
unsigned char**	CopyImage;					// This is the copy image
struct ImgProp 	ip;


void *Rotate(void* tid)
{
    long tn;            		     // My thread number (ID) is stored here
    int row,col,h,v,c;
	int NewRow,NewCol;
	double X, Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
    //struct Pixel pix;

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;
	
    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++){
        col=0;
        while(col<ip.Hpixels*3){
			// transpose image coordinates to Cartesian coordinates
			c=col/3;  		h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			X=(double)c-(double)h;
			Y=(double)v-(double)row;
			
			// pixel rotation matrix
			newX=cos(RotAngle)*X-sin(RotAngle)*Y;
			newY=sin(RotAngle)*X+cos(RotAngle)*Y;
			
			// Scale to fit everything in the image box
			H=(double)ip.Hpixels;
			V=(double)ip.Vpixels;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
				printf("ScaleFactor = %g\n", ScaleFactor);
			newX=newX*ScaleFactor;
			newY=newY*ScaleFactor;
			
			// convert back from Cartesian to image coordinates
			NewCol=((int) newX+h);
			NewRow=v-(int)newY;     
			if((NewCol>=0) && (NewRow>=0) && (NewCol<ip.Hpixels) && (NewRow<ip.Vpixels)){
				NewCol*=3;
				CopyImage[NewRow][NewCol]   = TheImage[row][col];
				CopyImage[NewRow][NewCol+1] = TheImage[row][col+1];
				CopyImage[NewRow][NewCol+2] = TheImage[row][col+2];
            }
            col+=3;
        }
    }
    pthread_exit(NULL);
}
void *Rotate2(void* tid)
{
    long tn;            		     // My thread number (ID) is stored here
    int row,col,h,v,c;
	int NewRow,NewCol;
	double X, Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
    //struct Pixel pix;

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;
	
			H=(double)ip.Hpixels;
			V=(double)ip.Vpixels;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++){
        col=0;
        while(col<ip.Hpixels*3){
			// transpose image coordinates to Cartesian coordinates
			c=col/3;  		h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			X=(double)c-(double)h;
			Y=(double)v-(double)row;
			
			// pixel rotation matrix
			newX=cos(RotAngle)*X-sin(RotAngle)*Y;
			newY=sin(RotAngle)*X+cos(RotAngle)*Y;
			
			// Scale to fit everything in the image box
//			H=(double)ip.Hpixels;
//			V=(double)ip.Vpixels;
//			Diagonal=sqrt(H*H+V*V);
//			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			newX=newX*ScaleFactor;
			newY=newY*ScaleFactor;
			
			// convert back from Cartesian to image coordinates
			NewCol=((int) newX+h);
			NewRow=v-(int)newY;     
			if((NewCol>=0) && (NewRow>=0) && (NewCol<ip.Hpixels) && (NewRow<ip.Vpixels)){
				NewCol*=3;
				CopyImage[NewRow][NewCol]   = TheImage[row][col];
				CopyImage[NewRow][NewCol+1] = TheImage[row][col+1];
				CopyImage[NewRow][NewCol+2] = TheImage[row][col+2];
            }
            col+=3;
        }
    }
    pthread_exit(NULL);
}


void *Rotate3(void* tid)
{
    long tn;            		     // My thread number (ID) is stored here
    int row,col,h,v,c;
	int NewRow,NewCol;
	double X, Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
	double CRA;
    //struct Pixel pix;

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;
	
			H=(double)ip.Hpixels;
			V=(double)ip.Vpixels;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			
			CRA=cos(RotAngle);
    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++){
        col=0;
        while(col<ip.Hpixels*3){
			// transpose image coordinates to Cartesian coordinates
			c=col/3;  		h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			X=(double)c-(double)h;
			Y=(double)v-(double)row;
			
			// pixel rotation matrix
			newX=CRA*X-sin(RotAngle)*Y;
			newY=sin(RotAngle)*X+CRA*Y;
//			newX=cos(RotAngle)*X-sin(RotAngle)*Y;
//			newY=sin(RotAngle)*X+cos(RotAngle)*Y;
			
			// Scale to fit everything in the image box
//			H=(double)ip.Hpixels;
//			V=(double)ip.Vpixels;
//			Diagonal=sqrt(H*H+V*V);
//			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			newX=newX*ScaleFactor;
			newY=newY*ScaleFactor;
			
			// convert back from Cartesian to image coordinates
			NewCol=((int) newX+h);
			NewRow=v-(int)newY;     
			if((NewCol>=0) && (NewRow>=0) && (NewCol<ip.Hpixels) && (NewRow<ip.Vpixels)){
				NewCol*=3;
				CopyImage[NewRow][NewCol]   = TheImage[row][col];
				CopyImage[NewRow][NewCol+1] = TheImage[row][col+1];
				CopyImage[NewRow][NewCol+2] = TheImage[row][col+2];
            }
            col+=3;
        }
    }
    pthread_exit(NULL);
}


void *Rotate4(void* tid)
{
    long tn;            		     // My thread number (ID) is stored here
    int row,col,h,v,c;
	int NewRow,NewCol;
	double X, Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
	double CRA,SRA;
    //struct Pixel pix;

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;
	
			H=(double)ip.Hpixels;
			V=(double)ip.Vpixels;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			
			CRA=cos(RotAngle);
			SRA=sin(RotAngle);
    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++){
        col=0;
        while(col<ip.Hpixels*3){
			// transpose image coordinates to Cartesian coordinates
			c=col/3;  		h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			X=(double)c-(double)h;
			Y=(double)v-(double)row;
			
			// pixel rotation matrix
			newX=CRA*X-SRA*Y;
			newY=SRA*X+CRA*Y;
//			newX=cos(RotAngle)*X-sin(RotAngle)*Y;
//			newY=sin(RotAngle)*X+cos(RotAngle)*Y;
			
			// Scale to fit everything in the image box
//			H=(double)ip.Hpixels;
//			V=(double)ip.Vpixels;
//			Diagonal=sqrt(H*H+V*V);
//			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			newX=newX*ScaleFactor;
			newY=newY*ScaleFactor;
			
			// convert back from Cartesian to image coordinates
			NewCol=((int) newX+h);
			NewRow=v-(int)newY;     
			if((NewCol>=0) && (NewRow>=0) && (NewCol<ip.Hpixels) && (NewRow<ip.Vpixels)){
				NewCol*=3;
				CopyImage[NewRow][NewCol]   = TheImage[row][col];
				CopyImage[NewRow][NewCol+1] = TheImage[row][col+1];
				CopyImage[NewRow][NewCol+2] = TheImage[row][col+2];
            }
            col+=3;
        }
    }
    pthread_exit(NULL);
}


void *Rotate5(void* tid)
{
    long tn;            		     // My thread number (ID) is stored here
//    int row,col,h,v,c;
    int row,col,h,v,c, hp3;
	int NewRow,NewCol;
	double X, Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
	double CRA,SRA;
    //struct Pixel pix;

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;
	
			H=(double)ip.Hpixels;
			V=(double)ip.Vpixels;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			
			CRA=cos(RotAngle);
			SRA=sin(RotAngle);
			h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			hp3=ip.Hpixels*3;
    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++){
        col=0;
		c=0;
//      while(col<ip.Hpixels*3){
        while(col<hp3){
			// transpose image coordinates to Cartesian coordinates
//			c=col/3;  		h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			X=(double)c-(double)h;
			Y=(double)v-(double)row;
			
			// pixel rotation matrix
			newX=CRA*X-SRA*Y;
			newY=SRA*X+CRA*Y;
//			newX=cos(RotAngle)*X-sin(RotAngle)*Y;
//			newY=sin(RotAngle)*X+cos(RotAngle)*Y;
			
			// Scale to fit everything in the image box
//			H=(double)ip.Hpixels;
//			V=(double)ip.Vpixels;
//			Diagonal=sqrt(H*H+V*V);
//			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			newX=newX*ScaleFactor;
			newY=newY*ScaleFactor;
			
			// convert back from Cartesian to image coordinates
			NewCol=((int) newX+h);
			NewRow=v-(int)newY;     
			if((NewCol>=0) && (NewRow>=0) && (NewCol<ip.Hpixels) && (NewRow<ip.Vpixels)){
				NewCol*=3;
				CopyImage[NewRow][NewCol]   = TheImage[row][col];
				CopyImage[NewRow][NewCol+1] = TheImage[row][col+1];
				CopyImage[NewRow][NewCol+2] = TheImage[row][col+2];
            }
			
            col+=3;
			c++;
        }
    }
    pthread_exit(NULL);
}


void *Rotate6(void* tid)
{
    long tn;            		     // My thread number (ID) is stored here
//    int row,col,h,v,c;
    int row,col,h,v,c, hp3;
	int NewRow,NewCol;
	double X, Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
	double CRA,SRA, CRAS, SRAS, SRAYS, CRAYS;
    //struct Pixel pix;

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;
	
			H=(double)ip.Hpixels;
			V=(double)ip.Vpixels;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			
			CRA=cos(RotAngle);	CRAS=ScaleFactor*CRA;
			SRA=sin(RotAngle);	SRAS=ScaleFactor*SRA;
			h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			hp3=ip.Hpixels*3;
    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++){
        col=0;
		c=0;
			Y=(double)v-(double)row;
			SRAYS=SRAS*Y;     CRAYS=CRAS*Y;
//      while(col<ip.Hpixels*3){
        while(col<hp3){
			// transpose image coordinates to Cartesian coordinates
//			c=col/3;  		h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			X=(double)c-(double)h;
//			Y=(double)v-(double)row;
			
			// pixel rotation matrix
			newX=CRAS*X-SRAYS;
			newY=SRAS*X+CRAYS;
//			newX=CRA*X-SRA*Y;
//			newY=SRA*X+CRA*Y;
//			newX=cos(RotAngle)*X-sin(RotAngle)*Y;
//			newY=sin(RotAngle)*X+cos(RotAngle)*Y;
			
			// Scale to fit everything in the image box
//			H=(double)ip.Hpixels;
//			V=(double)ip.Vpixels;
//			Diagonal=sqrt(H*H+V*V);
//			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
//			newX=newX*ScaleFactor;
//			newY=newY*ScaleFactor;
			
			// convert back from Cartesian to image coordinates
			NewCol=((int) newX+h);
			NewRow=v-(int)newY;     
			if((NewCol>=0) && (NewRow>=0) && (NewCol<ip.Hpixels) && (NewRow<ip.Vpixels)){
				NewCol*=3;
				CopyImage[NewRow][NewCol]   = TheImage[row][col];
				CopyImage[NewRow][NewCol+1] = TheImage[row][col+1];
				CopyImage[NewRow][NewCol+2] = TheImage[row][col+2];
            }
			
            col+=3;
			c++;
        }
    }
    pthread_exit(NULL);
}


void *Rotate7(void* tid)
{
    long tn;            		     // My thread number (ID) is stored here
//    int row,col,h,v,c;
    int row,col,h,v,hp3;
	double cc, ss, k1, k2;
	int NewRow,NewCol;
	double Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
	double CRA,SRA, CRAS, SRAS, SRAYS, CRAYS;
    //struct Pixel pix;

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;
	
			H=(double)ip.Hpixels;
			V=(double)ip.Vpixels;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			
			CRA=cos(RotAngle);	CRAS=ScaleFactor*CRA;
			SRA=sin(RotAngle);	SRAS=ScaleFactor*SRA;
			h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			hp3=ip.Hpixels*3;
    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++){
        col=0;
		cc=0.00;
		ss=0.00;
			Y=(double)v-(double)row;
			SRAYS=SRAS*Y;     CRAYS=CRAS*Y;
			k1=CRAS*(double)h + SRAYS;
			k2=SRAS*(double)h - CRAYS;
//      while(col<ip.Hpixels*3){
        while(col<hp3){
			// transpose image coordinates to Cartesian coordinates
//			c=col/3;  		h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
//			X=(double)c-(double)h;
//			Y=(double)v-(double)row;
			
			// pixel rotation matrix
			newX=cc-k1;
			newY=ss-k2;
//			newX=CRA*X-SRA*Y;
//			newY=SRA*X+CRA*Y;
//			newX=cos(RotAngle)*X-sin(RotAngle)*Y;
//			newY=sin(RotAngle)*X+cos(RotAngle)*Y;
			
			// Scale to fit everything in the image box
//			H=(double)ip.Hpixels;
//			V=(double)ip.Vpixels;
//			Diagonal=sqrt(H*H+V*V);
//			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
//			newX=newX*ScaleFactor;
//			newY=newY*ScaleFactor;
			
			// convert back from Cartesian to image coordinates
			NewCol=((int) newX+h);
			NewRow=v-(int)newY;     
			if((NewCol>=0) && (NewRow>=0) && (NewCol<ip.Hpixels) && (NewRow<ip.Vpixels)){
				NewCol*=3;
				CopyImage[NewRow][NewCol]   = TheImage[row][col];
				CopyImage[NewRow][NewCol+1] = TheImage[row][col+1];
				CopyImage[NewRow][NewCol+2] = TheImage[row][col+2];
            }
			
            col+=3;
			cc += CRAS;
			ss += SRAS;
        }
    }
    pthread_exit(NULL);
}
void *Rotate8(void* tid){
    long tn;            		     // My thread number (ID) is stored here
//    int row,col,h,v,c;
    int row,col,h,v,hp3;
	double cc, ss, k1, k2;
	int NewRow,NewCol;
	double Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
	double CRA,SRA, CRAS, SRAS, SRAYS, CRAYS;
    //struct Pixel pix;
	unsigned char Buffer [16384];

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;
	
			H=(double)ip.Hpixels;
			V=(double)ip.Vpixels;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			
			CRA=cos(RotAngle);	CRAS=ScaleFactor*CRA;
			SRA=sin(RotAngle);	SRAS=ScaleFactor*SRA;
			h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			hp3=ip.Hpixels*3;
    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++){
        memcpy((void *) Buffer, (void *) TheImage[row], (size_t) ip.Hbytes);
        col=0;
		cc=0.00;
		ss=0.00;
			Y=(double)v-(double)row;
			SRAYS=SRAS*Y;     CRAYS=CRAS*Y;
			k1=CRAS*(double)h + SRAYS;
			k2=SRAS*(double)h - CRAYS;
//      while(col<ip.Hpixels*3){
        while(col<hp3){
			// transpose image coordinates to Cartesian coordinates
//			c=col/3;  		h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
//			X=(double)c-(double)h;
//			Y=(double)v-(double)row;
			
			// pixel rotation matrix
			newX=cc-k1;
			newY=ss-k2;
//			newX=CRA*X-SRA*Y;
//			newY=SRA*X+CRA*Y;
//			newX=cos(RotAngle)*X-sin(RotAngle)*Y;
//			newY=sin(RotAngle)*X+cos(RotAngle)*Y;
			
			// Scale to fit everything in the image box
//			H=(double)ip.Hpixels;
//			V=(double)ip.Vpixels;
//			Diagonal=sqrt(H*H+V*V);
//			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
//			newX=newX*ScaleFactor;
//			newY=newY*ScaleFactor;
			
			// convert back from Cartesian to image coordinates
			NewCol=((int) newX+h);
			NewRow=v-(int)newY;     
			if((NewCol>=0) && (NewRow>=0) && (NewCol<ip.Hpixels) && (NewRow<ip.Vpixels)){
				NewCol*=3;
				CopyImage[NewRow][NewCol]   = Buffer[col];
				CopyImage[NewRow][NewCol+1] = Buffer[col+1];
				CopyImage[NewRow][NewCol+2] = Buffer[col+2];
            }
			
            col+=3;
			cc += CRAS;
			ss += SRAS;
        }
    }
    pthread_exit(NULL);
}

void *Rotate9(void* tid){
	//struct timeval 		t;
    //double         		StartTime, EndTime;
    //double         		TimeElapsed;
    //long tn;            		     // My thread number (ID) is stored here
//    int row,col,h,v,c;
    int row,col,h,v,hp3;
	double cc, ss, k1, k2;
	int NewRow,NewCol;
	double Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
	double CRA,SRA, CRAS, SRAS, SRAYS, CRAYS;
    //struct Pixel pix;
	unsigned char Buffer [16384];
	
	//gettimeofday(&t, NULL);
    //StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
    //tn = *((int *) tid);           // Calculate my Thread ID
    //tn *= ip.Vpixels/NumThreads;
	//read NextRowtoProcess
	//mutex stuff
	//actually process (should not be dependent on tid)
			H=(double)ip.Hpixels;
			V=(double)ip.Vpixels;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
			
			CRA=cos(RotAngle);	CRAS=ScaleFactor*CRA;
			SRA=sin(RotAngle);	SRAS=ScaleFactor*SRA;
			h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
			hp3=ip.Hpixels*3;
	while (NextRowtoProcess >= 0 && NextRowtoProcess < ip.Vpixels){
	pthread_mutex_lock(&nextrow_lock);
	row = NextRowtoProcess;
	NextRowtoProcess++;
	pthread_mutex_unlock(&nextrow_lock);

    //for(row=tn; row<tn+ip.Vpixels/NumThreads; row++){
        memcpy((void *) Buffer, (void *) TheImage[row], (size_t) ip.Hbytes);
        col=0;
		cc=0.00;
		ss=0.00;
			Y=(double)v-(double)row;
			SRAYS=SRAS*Y;     CRAYS=CRAS*Y;
			k1=CRAS*(double)h + SRAYS;
			k2=SRAS*(double)h - CRAYS;
//      while(col<ip.Hpixels*3){
        while(col<hp3){
			// transpose image coordinates to Cartesian coordinates
//			c=col/3;  		h=ip.Hpixels/2;   v=ip.Vpixels/2;	// integer div
//			X=(double)c-(double)h;
//			Y=(double)v-(double)row;
			
			// pixel rotation matrix
			newX=cc-k1;
			newY=ss-k2;
//			newX=CRA*X-SRA*Y;
//			newY=SRA*X+CRA*Y;
//			newX=cos(RotAngle)*X-sin(RotAngle)*Y;
//			newY=sin(RotAngle)*X+cos(RotAngle)*Y;
			
			// Scale to fit everything in the image box
//			H=(double)ip.Hpixels;
//			V=(double)ip.Vpixels;
//			Diagonal=sqrt(H*H+V*V);
//			ScaleFactor=(ip.Hpixels>ip.Vpixels) ? V/Diagonal : H/Diagonal;
//			newX=newX*ScaleFactor;
//			newY=newY*ScaleFactor;
			
			// convert back from Cartesian to image coordinates
			NewCol=((int) newX+h);
			NewRow=v-(int)newY;     
			if((NewCol>=0) && (NewRow>=0) && (NewCol<ip.Hpixels) && (NewRow<ip.Vpixels)){
				NewCol*=3;
				//pthread_mutex_lock(&image_lock);
				CopyImage[NewRow][NewCol]   = Buffer[col];
				CopyImage[NewRow][NewCol+1] = Buffer[col+1];
				CopyImage[NewRow][NewCol+2] = Buffer[col+2];
				//pthread_mutex_unlock(&image_lock);
            }
			
            col+=3;
			cc += CRAS;
			ss += SRAS;
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char** argv)
{
	int					RotDegrees, Function;
    int 				i,ThErr;
    struct timeval 		t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;
	char				FuncName[50];
	
    switch (argc){
		case 6 : NumThreads=atoi(argv[4]);  RotDegrees = atoi(argv[3]);		Function=atoi(argv[5]);//only possible input?
				 break;
		default: printf("\nUsage: %s inputBMP outputBMP [RotAngle] [NumThreads : 1-128] [Function:1-9]\n\n", argv[0]); //Wrong usage
				 printf("Example: %s infilename.bmp outname.bmp 125 4 8\n\n",argv[0]);				
				 printf("Nothing executed ... Exiting ...\n\n");
				 exit(EXIT_FAILURE);
    }
	if((NumThreads<1) || (NumThreads>MAXTHREADS)){
        printf("\nNumber of threads must be between 1 and %u... \n",MAXTHREADS);
        printf("\n'1' means Pthreads version with a single thread\n");
		printf("\n\nNothing executed ... Exiting ...\n\n");
        exit(EXIT_FAILURE);
	}
	if((RotDegrees<-360) || (RotDegrees>360)){
        printf("\nRotation angle of %d degrees is invalid ...\n",RotDegrees);
        printf("\nPlease enter an angle between -360 and +360 degrees ...\n");
		printf("\n\nNothing executed ... Exiting ...\n\n");
        exit(EXIT_FAILURE);
	}
	switch(Function){
		case 1:  strcpy(FuncName,"Rotate()");	RotateFunc=Rotate;	break;
		case 2:  strcpy(FuncName,"Rotate2()");	RotateFunc=Rotate2;	break;
		case 3:  strcpy(FuncName,"Rotate3()");	RotateFunc=Rotate3;	break;
		case 4:  strcpy(FuncName,"Rotate4()");	RotateFunc=Rotate4;	break;
		case 5:  strcpy(FuncName,"Rotate5()");	RotateFunc=Rotate5;	break;
		case 6:  strcpy(FuncName,"Rotate6()");	RotateFunc=Rotate6;	break;
		case 7:  strcpy(FuncName,"Rotate7() : Core-Friendly Version");	RotateFunc=Rotate7;	break;
		case 8:  strcpy(FuncName,"Rotate8() : Core-Friendly and Memory Friendly Version");	RotateFunc=Rotate8;	break;
		case 9:  strcpy(FuncName,"Rotate9() : Core-Friendly and Memory Friendly and Asynchronous Multithreading Version");	RotateFunc=Rotate9;	break;
		default: printf("Wrong function %d ... \n",Function);
					printf("\n\nNothing executed ... Exiting ...\n\n");
					exit(EXIT_FAILURE);
	}
	printf("\nLaunching %ld Pthread%s using function:  %s\n",NumThreads,NumThreads<=1?"":"s",FuncName);
	RotAngle=2*3.141592/360.000*(double) RotDegrees;   // Convert the angle to radians
	printf("\nRotating image by %d degrees (%5.4f radians) ...\n",RotDegrees,RotAngle);

	TheImage = ReadBMP(argv[1]);
	CopyImage = CreateBlankBMP(255);

	gettimeofday(&t, NULL);
    StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	
	pthread_attr_init(&ThAttr);
	pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);

		pthread_mutex_init(&image_lock, NULL);
		pthread_mutex_init(&nextrow_lock, NULL);
		for(i=0; i<NumThreads; i++){
			ThParam[i] = i;
			ThErr = pthread_create(&ThHandle[i], &ThAttr, RotateFunc, (void *)&ThParam[i]);
			if(ThErr != 0){
				printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
				exit(EXIT_FAILURE);
			}
		}
		for(i=0; i<NumThreads; i++){
			pthread_join(ThHandle[i], NULL);
		}
		pthread_mutex_destroy(&nextrow_lock);
		pthread_mutex_destroy(&image_lock);

	
    gettimeofday(&t, NULL);
    EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
	
    //merge with header and write to file
    WriteBMP(CopyImage, argv[2]);
 
 	// free() the allocated area for the images
	for(i = 0; i < ip.Vpixels; i++) { free(TheImage[i]); free(CopyImage[i]); }
	free(TheImage);   free(CopyImage);   
   
    printf("\n\nExecution time:%10.4f ms  ",TimeElapsed);
	if(NumThreads>=1) printf("(%10.4f  Thread-ms)  ",TimeElapsed*(double)NumThreads);
    printf(" (%6.3f ns/pixel)\n", 1000000*TimeElapsed/(double)(ip.Hpixels*ip.Vpixels));
    
    return (EXIT_SUCCESS);
}
