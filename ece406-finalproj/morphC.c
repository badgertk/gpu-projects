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
int 			Xo, Yo;						// Width and Height of strucuturing element
char			Type;						//
int 	     	ThParam[MAXTHREADS];		// Thread parameters ...
pthread_t      	ThHandle[MAXTHREADS];		// Thread handles
pthread_attr_t 	ThAttr;						// Pthread attrributes

unsigned char**	TheImage;					// This is the main image
unsigned char**	CopyImage;					// This is the copy image (to store edges)
double	**BWImage;					// B&W of TheImage (each pixel=double)
unsigned char	**DilationImage, **ErosionImage;				// Gauss filtered version of the B&W image
struct ImgProp 	ip;
unsigned char** StrElement;					// This is the structuring element

void *Dilation(void* tid)
{
	long tn;
	int row,col,i,j,max;
	
	tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;

	int d = Yo/2;
	int e = Xo/2;

	
    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++)
    {
		if((row<Yo) || (row>(ip.Vpixels-Yo -1))) continue;
        col=e;
        while(col<=(ip.Hpixels-Xo - 1)){
			max = 0;
			for(i=-d; i<=d; i++){
				for(j=-e; j<=e; j++){
					//if (max == 255) continue;
					if (StrElement[i+Yo/2][j+Xo/2] == 1){
						//printf("BWImage[%d][%d] = %d", row, col, (int)BWImage[row][col]);
						if  (max < (int) BWImage[row+i][col+j])
							max = (int) BWImage[row+i][col+j];
					}
				}
			}
			DilationImage[row][col*3]=max;
			DilationImage[row][col*3+1]=max;
			DilationImage[row][col*3+2]=max;
			//CopyImage[row][col*3+0]=max;
			//CopyImage[row][col*3+1]=max;
			//CopyImage[row][col*3+2]=max;
            col++;
        }
    }
    pthread_exit(NULL);
	
}

void *Erosion(void* tid)
{
	long tn;
	int row,col,i,j,min;
	
	tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;

	int d = Yo/2;
	int e = Xo/2;

	
    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++)
    {
		if((row<d) || (row>(ip.Vpixels-Yo -1))) continue;
        col=e;
        while(col<=(ip.Hpixels-Xo -1)){
				min = 255;
			for(i=-d; i<=d; i++){
				for(j=-e; j<=e; j++){
					//if (min == 0) continue;
					if (StrElement[i+Yo/2][j+Xo/2] == 1){
						if  (min > (int)BWImage[row+i][col+j])
							min = (int) BWImage[row+i][col+j];
					}
				}
			}
			ErosionImage[row][col*3]=min;
			ErosionImage[row][col*3+1]=min;
			ErosionImage[row][col*3+2]=min;
            col++;
        }
    }
    pthread_exit(NULL);
	
}

void *Difference(void* tid)
{
	long tn;
	int row,col;
	unsigned char PIXVAL;
	
	
	tn = *((int *) tid);		// Calculate my Thread ID
	tn *= ip.Vpixels/NumThreads;
	
	for (row = tn; row<tn+ip.Vpixels/NumThreads; row++)
	{
		col = 0;
		while(col<=(ip.Hpixels*3-1)){
			PIXVAL = DilationImage[row][col] - ErosionImage[row][col];
		//printf("CopyImage[%d][%d] = %c", row, col, PIXVAL);
			CopyImage[row][col] = PIXVAL;
		//CopyImage[row][col+1] = DilationImage[row][col] - ErosionImage[row][col];
		//CopyImage[row][col+2] = DilationImage[row][col] - ErosionImage[row][col];
		col++;// = col + 3;
		}
	}
	pthread_exit(NULL);
}

unsigned char** CreateBlankStrElement(unsigned char INIT){
    int i;//,j;

	unsigned char** img = (unsigned char **)malloc(Yo * sizeof(unsigned char*));
    for(i=0; i<Yo; i++){
        img[i] = (unsigned char *)malloc(Xo * sizeof(unsigned char));
		memset((void *)img[i],INIT,(size_t) Xo); // zero out every pixel
    }
    return img;
}

void CreateStrElement() {
	int i, j;
	if (Type=='S') {
		for (i = 0; i<Yo;i++) {
			for (j = 0; j < Xo; j++) {
				StrElement[i][j] = 1;
			}
		}
	}
	if (Type=='C') {
		int R = (int) sqrt(Xo*Xo/4+Yo*Yo/4);
		int H;
		for (i = -Yo/2; i<Yo/2; i++) {
			H = (int) sqrt(R*R-Yo*Yo/4);
			for (j = -H; j<H; j++) {
				StrElement[i+Yo/2][j+Xo/2] = 1;
			}
		}
	}
	if (Type=='X') {
		for (i = 0; i<Yo;i++) {
			if (i==Yo/2) {
				for (j = 0; j < Xo; j++) {
					StrElement[i][j] = 1;
				}
			}
		}
	}
/**
	IS THIS CORRECT???
	for (i = 0; i<Yo;i++) {
		for (j = 0; j < Xo; j++) {
			printf("StrElement[%d][%d] = %d", i, j, StrElement[i][j]);
		}
	}
**/
}

int main(int argc, char** argv) 
{
    int 				i,ThErr;
    struct timeval 		t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;
	//char				FuncName[50];
	
    switch (argc){
		case 7 : NumThreads=atoi(argv[6]);  Type = argv[3][0];	Xo = atoi(argv[4]);		Yo=atoi(argv[5]);//only possible input
				 break;
		default: printf("\nUsage: %s inputBMP outputBMP [S,X,C] [Width of StrEl] [Height of StrEl] [NumThreads : 1-128]\n\n", argv[0]); //Wrong usage
				 printf("Where: S is square-shaped StrEl, X is cross-shaped StrEl, and C is circular StrEl.\n");
				 printf("Example: %s infilename.bmp outname.bmp C 4 4 8\n\n",argv[0]);				
				 printf("Nothing executed ... Exiting ...\n\n");
				 exit(EXIT_FAILURE);
    }

	if((NumThreads<1) || (NumThreads>MAXTHREADS)){
        printf("\nNumber of threads must be between 1 and %u... \n",MAXTHREADS);
        printf("\n'1' means Pthreads version with a single thread\n");
		printf("\n\nNothing executed ... Exiting ...\n\n");
        exit(EXIT_FAILURE);
	}
	switch(Type){
		case 'C':  	
			if(Xo != Yo){
				printf("Error: For circles, StrEl width must equal StrEl height.\n");
				exit(EXIT_FAILURE);
			}
			break;
		case 'S': break;
		case 'X': break;
		default: printf("Wrong type %d ... \n",Type);
					printf("\n\nNothing executed ... Exiting ...\n\n");
					exit(EXIT_FAILURE);
	}
	//printf("\nLaunching %ld Pthread%s using function:  %s\n",NumThreads,NumThreads<=1?"":"s",FuncName);
	//RotAngle=2*3.141592/360.000*(double) RotDegrees;   // Convert the angle to radians
	//printf("\nRotating image by %d degrees (%5.4f radians) ...\n",RotDegrees,RotAngle);

	TheImage = ReadBMP(argv[1]);
	CopyImage = CreateBlankBMP(255);		// This will store the edges in RGB  
	BWImage    = CreateBWCopy(TheImage);
	DilationImage = CreateBlankBMP(255);
	ErosionImage = CreateBlankBMP(255);
	StrElement = CreateBlankStrElement(0);
	CreateStrElement();
	
	gettimeofday(&t, NULL);
    StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	
	pthread_attr_init(&ThAttr);
	pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);
	
	for(i=0; i<NumThreads; i++){
		ThParam[i] = i;
		ThErr = pthread_create(&ThHandle[i], &ThAttr, Dilation, (void *)&ThParam[i]);
		if(ThErr != 0){
			printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
			exit(EXIT_FAILURE);
		}
	}
	for(i=0; i<NumThreads; i++){
		pthread_join(ThHandle[i], NULL);
	}
	for(i=0; i<NumThreads; i++){
		ThParam[i] = i;
		ThErr = pthread_create(&ThHandle[i], &ThAttr, Erosion, (void *)&ThParam[i]);
		if(ThErr != 0){
			printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
			exit(EXIT_FAILURE);
		}
	}
	for(i=0; i<NumThreads; i++){
		pthread_join(ThHandle[i], NULL);
	}
	/**
	int j,k;
	for (k = 0; k < ip.Vpixels; k++){
		for (j = 0; j < ip.Hpixels; j++){
			printf("Dilation[][] = %d and Erosion[][] = %d\n", DilationImage[k][j], ErosionImage[k][j]);
		}
	}**/
	for(i=0; i<NumThreads; i++){
		ThParam[i] = i;
		ThErr = pthread_create(&ThHandle[i], &ThAttr, Difference, (void *)&ThParam[i]);
		if(ThErr != 0){
			printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
			exit(EXIT_FAILURE);
		}
	}
		pthread_attr_destroy(&ThAttr);
	for(i=0; i<NumThreads; i++){
		pthread_join(ThHandle[i], NULL);
	}
	
    gettimeofday(&t, NULL);
    EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
	
    //merge with header and write to file
    WriteBMP(CopyImage, argv[2]); 
   
    printf("\n\nExecution time:%10.4f ms  ",TimeElapsed);
	if(NumThreads>=1) printf("(%10.4f  Thread-ms)  ",TimeElapsed*(double)NumThreads);
    printf(" (%6.3f ns/pixel)\n", 1000000*TimeElapsed/(double)(ip.Hpixels*ip.Vpixels));
	
	// free() the allocated area for image and pointers
	for(i = 0; i < ip.Vpixels; i++) { 
		free(TheImage[i]);   free(CopyImage[i]); free(BWImage[i]); 
		free(DilationImage[i]); free(ErosionImage[i]);   
	}
	for (i = 0; i < Xo; i++){
		free(StrElement[i]);
	}
	free(TheImage);  free(CopyImage);  free(BWImage);  
	free(DilationImage);  free(ErosionImage);  free(StrElement);
    
    return (EXIT_SUCCESS);
}
