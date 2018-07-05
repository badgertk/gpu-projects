imflipC	:	imflipC.c ImageStuff.c ImageStuff.h
				gcc imflipC.c ImageStuff.c -o imflipC -lpthread

imflipG 	:	imflipG.cu
				nvcc imflipG.cu -o imflipG -lopencv_core -lopencv_highgui

imflipCL	:	imflipCL.cpp
				g++ imflipCL.cpp -o imflipCL -lOpenCL -lopencv_core -lopencv_highgui


# REMINDER: type these module loads before trying to compile
#			module load gtk
#			module load gstreamer/0.10.36/b1
#			module load opencv
#			module load cuda
#			module load gcc/4.7.2
