imrotate	:	imrotate.c ImageStuff.c ImageStuff.h
				gcc imrotate.c ImageStuff.c -o imrotate -lpthread -lm

imrotateMC	:	imrotateMC.c ImageStuff.c ImageStuff.h
				gcc imrotateMC.c ImageStuff.c -o imrotateMC -lpthread -lm
				
imrotateG	:	imrotateG.cu
				nvcc imrotateG.cu -o imrotateG -lopencv_core -lopencv_highgui
        
imrotateCL	:	imrotateCL.cpp
				g++ imrotateCL.cpp -o imrotateCL -lOpenCL -lopencv_core -lopencv_highgui
