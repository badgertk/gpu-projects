.PHONY : all, clean

all		:	imflip imflipG

imflip		:	imflip.c ImageStuff.c ImageStuff.h
				gcc imflip.c ImageStuff.c -o imflip

imflipG		:	opencv_lab.cu
				nvcc opencv_lab.cu -o imflipG -lopencv_core -lopencv_highgui

clean		:
			rm -f imflip imflipG
