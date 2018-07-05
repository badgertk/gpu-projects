#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "ImageStuff.h"


unsigned char** ReadBMP(char* filename)
{
	int i;
	FILE* f = fopen(filename, "rb");
	if(f == NULL)
	{
		printf("\n\n%s NOT FOUND\n\n",filename);
		exit(1);
	}

	unsigned char HeaderInfo[54];
	fread(HeaderInfo, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];
	int height = *(int*)&HeaderInfo[22];

	//copy header for re-use
	for(i=0; i<54; i++) {
		ip.HeaderInfo[i] = HeaderInfo[i];
	}

	ip.Vpixels = height;
	ip.Hpixels = width;
	int RowBytes = (width*3 + 3) & (~3);
	ip.Hbytes = RowBytes;

	printf("\n   Input BMP File name: %20s  (%u x %u)",filename,ip.Hpixels,ip.Vpixels);

	unsigned char tmp;
	unsigned char **TheImage = (unsigned char **)malloc(height * sizeof(unsigned char*));
	for(i=0; i<height; i++) {
		TheImage[i] = (unsigned char *)malloc(RowBytes * sizeof(unsigned char));
	}

	for(i = 0; i < height; i++) {
		fread(TheImage[i], sizeof(unsigned char), RowBytes, f);
	}


	fclose(f);
	return TheImage;  // remember to free() it in caller!
}

void WriteBMP(unsigned char** img, char* filename)
{
	FILE* f = fopen(filename, "wb");
	if(f == NULL)
	{
		printf("\n\nFILE CREATION ERROR: %s\n\n",filename);
		exit(1);
	}

	unsigned long int x,y;
	char temp;
	int filesize = ip.Hbytes*ip.Vpixels+54;
	int datasize = filesize - 54;

	//write header
	for(x=0; x<2; x++) {
		fputc(ip.HeaderInfo[x],f);
	}
	fwrite( &filesize, 4, 1, f );  // size of the BMP file in bytes
	for(x=6; x<18; x++) {
		fputc(ip.HeaderInfo[x],f);
	}
	fwrite( &ip.Hpixels, 4, 1, f );  // width in px
	fwrite( &ip.Vpixels, 4, 1, f );  // height in px
	for(x=26; x<34; x++) {
		fputc(ip.HeaderInfo[x],f);
	}
	fwrite( &datasize, 4, 1, f );  // size of raw bitmap data
	for(x=38; x<54; x++) {
		fputc(ip.HeaderInfo[x],f);
	}
	// TODO: make sure all fwrite's are doing little endian

	//write data
	for(x=0; x<ip.Vpixels; x++) {
		for(y=0; y<ip.Hbytes; y++)
		{
			temp=img[x][y];
			fputc(temp,f);
		}
	}
	printf("\n  Output BMP File name: %20s  (%u x %u)",filename,ip.Hpixels,ip.Vpixels);

	fclose(f);
}
