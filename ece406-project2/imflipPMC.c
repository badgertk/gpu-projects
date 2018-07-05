#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>

#include "ImageStuff.h"

#define REPS 	     1
#define MAXTHREADS   128

long  			NumThreads;         		// Total number of threads working in parallel
int 	     	ThParam[MAXTHREADS];		// Thread parameters ...
pthread_t      	ThHandle[MAXTHREADS];		// Thread handles
pthread_attr_t 	ThAttr;						// Pthread attrributes
void (*FlipFunc)(unsigned char** img);		// Function pointer to flip the image
void* (*MTFlipFunc)(void *arg);				// Function pointer to flip the image, multi-threaded version

unsigned char**	TheImage;					// This is the main image
//unsigned char** NewImage;					// This is the new image
struct ImgProp 	ip;

//FILE *fs;
//FILE *ft;


void FlipImageV(unsigned char** img)
{
    struct Pixel pix; //temp swap pixel
    int row, col;
    
    //vertical flip
    for(col=0; col<ip.Hbytes; col+=3)
    {
        row = 0;
        while(row<ip.Vpixels/2)
        {
            pix.B = img[row][col];
            pix.G = img[row][col+1];
            pix.R = img[row][col+2];
            
            img[row][col]   = img[ip.Vpixels-(row+1)][col];
            img[row][col+1] = img[ip.Vpixels-(row+1)][col+1];
            img[row][col+2] = img[ip.Vpixels-(row+1)][col+2];
            
            img[ip.Vpixels-(row+1)][col]   = pix.B;
            img[ip.Vpixels-(row+1)][col+1] = pix.G;
            img[ip.Vpixels-(row+1)][col+2] = pix.R;
            
            row++;
        }
    }
}


void FlipImageH(unsigned char** img)
{
    struct Pixel pix; //temp swap pixel
    int row, col;
    
    //horizontal flip
    for(row=0; row<ip.Vpixels; row++)
    {
        col = 0;
        while(col<(ip.Hpixels*3)/2)
        {
            pix.B = img[row][col];
            pix.G = img[row][col+1];
            pix.R = img[row][col+2];
            
            img[row][col]   = img[row][ip.Hpixels*3-(col+3)];
            img[row][col+1] = img[row][ip.Hpixels*3-(col+2)];
            img[row][col+2] = img[row][ip.Hpixels*3-(col+1)];
            
            img[row][ip.Hpixels*3-(col+3)] = pix.B;
            img[row][ip.Hpixels*3-(col+2)] = pix.G;
            img[row][ip.Hpixels*3-(col+1)] = pix.R;
            
            col+=3;
        }
    }
}


void *MTFlipV(void* tid)
{
    struct Pixel pix; //temp swap pixel
    int row, col;

    long ts = *((int *) tid);       	// My thread ID is stored here
    ts *= ip.Hbytes/NumThreads;			// start index
	long te = ts+ip.Hbytes/NumThreads-1; 	// end index

    for(col=ts; col<=te; col+=3)
    {
        row=0;
        while(row<ip.Vpixels/2)
        {
            pix.B = TheImage[row][col];
            pix.G = TheImage[row][col+1];
            pix.R = TheImage[row][col+2];
            
            TheImage[row][col]   = TheImage[ip.Vpixels-(row+1)][col];
            TheImage[row][col+1] = TheImage[ip.Vpixels-(row+1)][col+1];
            TheImage[row][col+2] = TheImage[ip.Vpixels-(row+1)][col+2];
            
            TheImage[ip.Vpixels-(row+1)][col]   = pix.B;
            TheImage[ip.Vpixels-(row+1)][col+1] = pix.G;
            TheImage[ip.Vpixels-(row+1)][col+2] = pix.R;
            
            row++;
        }
    }
    pthread_exit(NULL);
}


void *MTFlipH(void* tid)
{
    struct Pixel pix; //temp swap pixel
    int row, col;

    long ts = *((int *) tid);       	// My thread ID is stored here
    ts *= ip.Vpixels/NumThreads;			// start index
	long te = ts+ip.Vpixels/NumThreads-1; 	// end index

    for(row=ts; row<=te; row++){
        col=0;
        while(col<ip.Hpixels*3/2){
            pix.B = TheImage[row][col];
            pix.G = TheImage[row][col+1];
            pix.R = TheImage[row][col+2];
            
            TheImage[row][col]   = TheImage[row][ip.Hpixels*3-(col+3)];
            TheImage[row][col+1] = TheImage[row][ip.Hpixels*3-(col+2)];
            TheImage[row][col+2] = TheImage[row][ip.Hpixels*3-(col+1)];
            
            TheImage[row][ip.Hpixels*3-(col+3)] = pix.B;
            TheImage[row][ip.Hpixels*3-(col+2)] = pix.G;
            TheImage[row][ip.Hpixels*3-(col+1)] = pix.R;
            
            col+=3;
        }
    }
    pthread_exit(NULL);
}


void *MTFlipHM(void* tid)
{
    struct Pixel pix; //temp swap pixel
    int row, col;
	unsigned char Buffer[16384];	 // This is the buffer to use to get the entire row

    long ts = *((int *) tid);       	// My thread ID is stored here
    ts *= ip.Vpixels/NumThreads;			// start index
	long te = ts+ip.Vpixels/NumThreads-1; 	// end index

    for(row=ts; row<=te; row++){
        memcpy((void *) Buffer, (void *) TheImage[row], (size_t) ip.Hbytes);
		col=0;
        while(col<ip.Hpixels*3/2){
            pix.B = Buffer[col];
            pix.G = Buffer[col+1];
            pix.R = Buffer[col+2];
            
            Buffer[col]   = Buffer[ip.Hpixels*3-(col+3)];
            Buffer[col+1] = Buffer[ip.Hpixels*3-(col+2)];
            Buffer[col+2] = Buffer[ip.Hpixels*3-(col+1)];
            
            Buffer[ip.Hpixels*3-(col+3)] = pix.B;
            Buffer[ip.Hpixels*3-(col+2)] = pix.G;
            Buffer[ip.Hpixels*3-(col+1)] = pix.R;
            
            col+=3;
        }
        memcpy((void *) TheImage[row], (void *) Buffer, (size_t) ip.Hbytes);
    }
    pthread_exit(NULL);
}

void *MTFlipHMC(void* tid){
    int row, i, j, k;

    long ts = *((int *) tid);       	// My thread ID is stored here
    ts *= ip.Vpixels/NumThreads;			// start index
	long te = ts+ip.Vpixels/NumThreads-1; 	// end index
	//printf("ts = %ld te = %ld", ts, te);
				//printf("ip.Hpixels*3 = %d", ip.Hpixels*3); //9600
				//printf("ip.Hpixels*3 >> 1 = %d", ip.Hpixels*3 >> 1); //4800
				//printf("size_t 24 = %d", ((size_t) 24)); //24
	unsigned long Snippet;
	unsigned long Snippet2;
	unsigned long SnippetResult;
	unsigned long SnippetResult2;
	unsigned char Chunk[24]; //24 byte chunk read from image
	unsigned char Chunk2[24];
	unsigned char ChunkResult[24]; //24 byte chunk appropriately reflected
	unsigned char ChunkResult2[24];
	unsigned long Mask = 0xFFFFFF;
	unsigned long Mask2 = 0xFF;

    for(row=ts; row<=te; row++){
		int NumChunks = ip.Hpixels >> 3;  //3200 pixels across 8 pixels per chunk = 400 chunks per line
			for (j = 0; j< NumChunks >> 1; j++){ //go halfway

			//memory efficient data loads
			memcpy((void *) Chunk, (void *) TheImage[row]+j*24, 8);
			memcpy((void *) Chunk + 8, (void *) TheImage[row]+j*24 + 8, 8);
			memcpy((void *) Chunk + 16, (void *) TheImage[row]+j*24+ 16, 8);
			memcpy((void *) Chunk2, (void *) TheImage[row] + ip.Hpixels*3 - j*24 - 24 , 8);

			memcpy((void *) Chunk2 + 8, (void *) TheImage[row] + ip.Hpixels*3 - j*24 - 16 , 8);
			memcpy((void *) Chunk2 + 16, (void *) TheImage[row] + ip.Hpixels*3 - j*24 - 8 , 8);
/*
			 if (row == 0 && j == 0){
			printf("\nj = %d\nChunk is = ", j);
			for (i = 0; i < 24; i++){
				printf(" %d ", Chunk[i]);
			}
			printf("\n");
			printf("\nj = %d\nChunk2 is = ", j);
			for (i = 0; i < 24; i++){
				printf(" %d ", Chunk2[i]);
			}
			printf("\n");
			} */
			
			//printf("Segmentation fault under here?");
			//Chunk -> Snippet
			//int NumSnippet = 4;
 			//if (row == 0){
			for (k = 0; k < 4; k++){
				Snippet = 0; //snippet of RGB values contains two pixels worth
				Snippet2 = 0;
				SnippetResult = 0;
				SnippetResult2 = 0;
				for (i = 0; i< 6; i++){
					Snippet = Snippet + Chunk[6*k+i];
					Snippet2 = Snippet2 + Chunk2[6*k+i];
					//printf("Snippet = %lx + Chunk[%d] = %d\n", Snippet, i, Chunk[i]);
					Snippet2 = Snippet2 << 8;
					Snippet = Snippet << 8;
					//printf("Shifted Snippet = %lx\n", Snippet);
				}
				Snippet2 = Snippet2 >> 8;
				Snippet = Snippet >> 8;
				//Snippet [A B] -> Snippet [B A]
				//printf("Snippet = %lu          ", Snippet);
				SnippetResult = Snippet  & Mask; //[A B] & [0 1] = [0 B]
				SnippetResult2 = Snippet2  & Mask; //[A B] & [0 1] = [0 B]
				SnippetResult = SnippetResult << 8*3; // [0 B] -> [B 0]
				SnippetResult2 = SnippetResult2 << 8*3; // [0 B] -> [B 0]
				Snippet = Snippet >> 8*3; //[A B] -> [0 A]
				Snippet2 = Snippet2 >> 8*3; //[A B] -> [0 A]
				SnippetResult = SnippetResult + (Snippet & Mask); //[0 A] & [0 1] = [B A]
				SnippetResult2 = SnippetResult2 + (Snippet2 & Mask); //[0 A] & [0 1] = [B A]
				//printf("SnippetResult = %lu\n", SnippetResult);
				//Snippet -> Chunk
				for (i = 0; i< 6; i++){
					ChunkResult[24-(6*k+i)-1] = SnippetResult & Mask2;
					ChunkResult2[24-(6*k+i)-1] = SnippetResult2 & Mask2;
					//printf("ChunkResult[%d] = %lx\n", i, SnippetResult & Mask2);
					SnippetResult = SnippetResult >> 8;
					SnippetResult2 = SnippetResult2 >> 8;
					//printf("Shifted SnippetResult = %lx\n", SnippetResult);
				}

			}
			//}
/*			if (row == 0 && j == 0){
 			printf("\nj = %d\nChunkResult is = ", j);
			for (i = 0; i < 24; i++){
				printf(" %d ", ChunkResult[i]);
			}
			printf("\n"); 
			printf("\nj = %d\nChunkResult2 is = ", j);
			for (i = 0; i < 24; i++){
				printf(" %d ", ChunkResult2[i]);
			} 
			printf("\n");
			}*/
			//memory efficient data stores
			//the value after TheImage[row]+ seem incorrect.
 			memcpy((void *) TheImage[row]+j*24, (void *) ChunkResult2, 8);
			memcpy((void *) TheImage[row]+j*24 + 8, (void *) ChunkResult2 + 8, 8);
			memcpy((void *) TheImage[row]+j*24+ 16, (void *) ChunkResult2 + 16, 8);
			memcpy((void *) TheImage[row] + ip.Hpixels*3 - j*24 - 24, (void *) ChunkResult, 8);
			memcpy((void *) TheImage[row] + ip.Hpixels*3 - j*24 - 16, (void *) ChunkResult + 8, 8);
			memcpy((void *) TheImage[row] + ip.Hpixels*3 - j*24 - 8, (void *) ChunkResult + 16, 8); 
		}
    }
	
	pthread_exit(NULL); //point of termination
}


void *MTFlipVM(void* tid)
{
    //struct Pixel pix; //temp swap pixel
    int row, row2;//, col;
	unsigned char Buffer[16384];	 // This is the buffer to get the first row
	unsigned char Buffer2[16384];	 // This is the buffer to get the second row

    long ts = *((int *) tid);       	// My thread ID is stored here
    ts *= ip.Vpixels/NumThreads/2;				// start index
	long te = ts+(ip.Vpixels/NumThreads/2)-1; 	// end index

    for(row=ts; row<=te; row++){
        memcpy((void *) Buffer, (void *) TheImage[row], (size_t) ip.Hbytes);
        row2=ip.Vpixels-(row+1);   
		memcpy((void *) Buffer2, (void *) TheImage[row2], (size_t) ip.Hbytes);
		// swap row with row2
		memcpy((void *) TheImage[row], (void *) Buffer2, (size_t) ip.Hbytes);
		memcpy((void *) TheImage[row2], (void *) Buffer, (size_t) ip.Hbytes);
    }
    pthread_exit(NULL);
}

void *MTFlipVM2(void* tid){ //done
    int row, row2;//, col;
	unsigned char Buffer[16384*2];	 // This is the buffer to get the first row

    long ts = *((int *) tid);       	// My thread ID is stored here
    ts *= ip.Vpixels/NumThreads/2;				// start index
	long te = ts+(ip.Vpixels/NumThreads/2)-1; 	// end index

    for(row=ts; row<=te; row++){
        memcpy((void *) Buffer, (void *) TheImage[row], (size_t) ip.Hbytes);
        row2=ip.Vpixels-(row+1);   
		memcpy((void *) TheImage[row], (void *) TheImage[row2], (size_t) ip.Hbytes);
		memcpy((void *) TheImage[row2], (void *) Buffer, (size_t) ip.Hbytes);
    }
    pthread_exit(NULL);
}

void *MTFlipVM3(void* threadID){
	int simulrow = 2; //how many rows are simultaneously processed
	int i = 0;
    int row, row2;//, col;
	unsigned char Buffer[16384*simulrow];	 // This is the buffer to get the first row
	unsigned char Buffer2[16384*simulrow];	 // This is the buffer to get the second row
	

    long tid = *((int *) threadID);       	// My thread ID is stored here
	long ts;
	long te;
/* 	if (ip.Vpixels % NumThreads != 0){
		ts = tid * (ip.Vpixels/NumThreads/2 + 1);
			if (tid == (NumThreads - 1)){ //last thread will be assigned fewer elements
				te = ip.Vpixels/2 - 1;
			}
			else {
				te = ts + ip.Vpixels/NumThreads/2-1;
			}
	}
	else {
		ts = tid * ip.Vpixels/NumThreads/2;
		te = ts + (ip.Vpixels/NumThreads/2)-1;
	} */
    ts = tid * ip.Vpixels/NumThreads/2;		// start index
	te = ts+(ip.Vpixels/NumThreads/2)-1; 	// end index

     for(row=ts; row<=te - simulrow + 1; row = row + simulrow){ //read first row
        memcpy((void *) Buffer, (void *) TheImage[row], (size_t) ip.Hbytes + 4 * simulrow + 1);
		for (i = 1; i < simulrow; i++){ //read consecutive rows
			memcpy((void *) Buffer+i*(ip.Hbytes + 4 * simulrow + 1), (void *) TheImage[row] + i*(ip.Hbytes + 4 * simulrow + 1), (size_t) ip.Hbytes + 4 * simulrow + 1);
		}

         row2=ip.Vpixels-(row+simulrow);   
			memcpy((void *) Buffer2, (void *) TheImage[row2-(simulrow - 1)], (size_t) ip.Hbytes + 4 * simulrow + 1);
		for (i = 1; i < simulrow; i++){
			memcpy((void *) Buffer2+i*(ip.Hbytes + 4 * simulrow + 1), (void *) TheImage[row2 - (simulrow - 1)]+ i*(ip.Hbytes + 4 * simulrow + 1), (size_t) ip.Hbytes + 4 * simulrow + 1);
		}

		// swap row with row2
		memcpy((void *) TheImage[row], (void *) Buffer2, (size_t) (ip.Hbytes + 4 * simulrow + 1)*simulrow);
		memcpy((void *) TheImage[row2-(simulrow - 1)], (void *) Buffer, (size_t) (ip.Hbytes + 4 * simulrow + 1)*simulrow); 
    } 

    pthread_exit(NULL);
}

int main(int argc, char** argv)
{
	//char 				Flip;
    int 				a,i,ThErr, version;
    struct timeval 		t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;
	char				FlipType[50];
	
    switch (argc){
		case 5: NumThreads = atoi(argv[3]);
				version = atoi(argv[4]);//version 0 - 4
				break;
/* 		default: printf(); //usage not right
		case 3 : NumThreads=0; 				Flip = 'V';						break;
		case 4 : NumThreads=0;  			Flip = toupper(argv[3][0]);		break;
		case 5 : NumThreads=atoi(argv[4]);  Flip = toupper(argv[3][0]);		break; */
		default: printf("\n\nUsage: imflipPMC infile.bmp outfile.bmp numThreads [version]\n");
				 printf("Use 0 for the memory-friendly version of the horizontal flip program.  Use 1 for the memory and core-friendly version of the horizontal flip program.\n");
				 printf("Use 2 for the memory-friendly version of the vertical flip program.  Use 3 for the multiple row memory access version of the veritcal flip program.  Use 4 for the baseline version of the vertical flip program.\n");
				/*  printf("\n\nUse 'V', 'H' for regular, and 'W', 'I' for the memory-friendly version of the program\n\n"); */
				 printf("\n\nNumThreads=0 for the serial version, and 1-128 for the Pthreads version\n\n");
				 printf("\n\nExample: imflipPMC infilename.bmp outname.bmp 8 0\n\n");
				 printf("\n\nExample: imflipPMC infilename.bmp outname.bmp 0 1\n\n");
				 printf("\n\nNothing executed ... Exiting ...\n\n");
				exit(EXIT_FAILURE);
    }
	switch (version){
		case 0:	 MTFlipFunc = MTFlipHM; FlipFunc=FlipImageH;	strcpy(FlipType,"Horizontal"); 
				 printf("\nExecuting the Memory-friendly version.\n");
				 break;
		case 1:	 MTFlipFunc = MTFlipHMC; FlipFunc=FlipImageH;	strcpy(FlipType,"Horizontal"); 
				 printf("\nExecuting the Memory and Core-friendly version.\n");
				 break;
		case 2:	 MTFlipFunc = MTFlipVM2; FlipFunc=FlipImageV;	strcpy(FlipType,"Vertical"); 
				 break;
		case 3:	 MTFlipFunc = MTFlipVM3; FlipFunc=FlipImageV;	strcpy(FlipType,"Vertical"); 
				 break;
		case 4:  MTFlipFunc = MTFlipVM;	 FlipFunc=FlipImageV;	strcpy(FlipType,"Vertical");
				 break;
		default: printf("not proper usage\n"); 
				 exit(EXIT_FAILURE);//not proper usage
	}

	if((NumThreads<0) || (NumThreads>MAXTHREADS)){
            printf("\nNumber of threads must be between 0 and %u... \n",MAXTHREADS);
            printf("\n'1' means Pthreads version with a single thread\n");
            printf("\nYou can also specify '0' which means the 'serial' (non-Pthreads) version... \n\n");
			 printf("\n\nNothing executed ... Exiting ...\n\n");
            exit(EXIT_FAILURE);
	}
	if(NumThreads == 0){
		printf("\nExecuting the serial (non-Pthreaded) version ...\n");
	}else{
		printf("\nExecuting the multi-threaded version with %ld threads ...\n",NumThreads);
	}
/* 
	switch(Flip){
		case 'V' : 	MTFlipFunc = MTFlipV;  FlipFunc=FlipImageV; strcpy(FlipType,"Vertical (V)"); break;
		case 'H' : 	MTFlipFunc = MTFlipH;  FlipFunc=FlipImageH; strcpy(FlipType,"Horizontal (H)"); break;
		case 'W' : 	MTFlipFunc = MTFlipVM; FlipFunc=FlipImageV; strcpy(FlipType,"Vertical (W)"); break;
		case 'I' : 	MTFlipFunc = MTFlipHM; FlipFunc=FlipImageH; strcpy(FlipType,"Horizontal (I)"); break;
		default  : 	printf("Flip option '%c' is invalid. Can only be 'V', 'H', 'W', or 'I'\n",Flip);
					printf("\n\nNothing executed ... Exiting ...\n\n");
					exit(EXIT_FAILURE);
	} */

	TheImage = ReadBMP(argv[1]);
/* 	fs = fopen(argv[3], "w");
	for (i = 0; i < ip.Hpixels*3; i = i + 3){
	fprintf(fs,"|PIXEL %d ->| %d %d %d ", i/3, TheImage[0][i], TheImage[0][i+1], TheImage[0][i+2]);
	
	}
		fclose(fs); //close the file  */
/* 	printf("The first 24 bytes are \n");
	for (i = 0; i < ip.Hbytes*3; i++){
	printf(" %d ", TheImage[0][i]);
}
	printf("\n"); */
	gettimeofday(&t, NULL);
    StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	
    if(NumThreads >0){
		pthread_attr_init(&ThAttr);
		pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);
		for(a=0; a<REPS; a++){
			for(i=0; i<NumThreads; i++){
				ThParam[i] = i;
				ThErr = pthread_create(&ThHandle[i], &ThAttr, MTFlipFunc, (void *)&ThParam[i]);
				if(ThErr != 0){
					printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
					exit(EXIT_FAILURE);
				}
			}
			for(i=0; i<NumThreads; i++){
				pthread_join(ThHandle[i], NULL);
			}
		}
	}else{
		for(a=0; a<REPS; a++){
			(*FlipFunc)(TheImage);
		}
	}
	
    gettimeofday(&t, NULL);
    EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
	TimeElapsed/=(double)REPS;
	
    //merge with header and write to file
    WriteBMP(TheImage, argv[2]);
	
/* 	ft = fopen(argv[4], "w");
	for (i = 0; i < ip.Hpixels*3; i = i + 3){
	fprintf(ft,"| %d %d %d ", TheImage[0][i], TheImage[0][i+1], TheImage[0][i+2]);
	
	}
		fclose(ft); //close the file  */
	
 	// free() the allocated memory for the image
	for(i = 0; i < ip.Vpixels; i++) { free(TheImage[i]);  /*free(NewImage[i]);*/}
	free(TheImage);
	//free(NewImage);
   
    printf("\n\nTotal execution time: %9.4f ms.  ",TimeElapsed);
	if(NumThreads>1) printf("(%9.4f ms per thread).  ",TimeElapsed/(double)NumThreads);
	printf("\n\nFlip Type =  '%s'",FlipType);
    printf("\n (%6.3f ns/pixel)\n", 1000000*TimeElapsed/(double)(ip.Hpixels*ip.Vpixels));
    
    return (EXIT_SUCCESS);
}
