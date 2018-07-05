.PHONY : all, clean

all		:	imedgeG

imedge		:	imedge.c ImageStuff.c ImageStuff.h
				gcc imedge.c ImageStuff.c -o imedge -lpthread -lm
				
imedgeMC	:	imedgeMC.c ImageStuff.c ImageStuff.h
				gcc imedgeMC.c ImageStuff.c -o imedgeMC -lpthread -lm
				
imedgeMCT	:	imedgeMCT.c ImageStuff.c ImageStuff.h
				gcc imedgeMCT.c ImageStuff.c -o imedgeMCT -lpthread -lm
				
imedgeGStr	:	imedgeGStr.cu
			#nvcc -g -G -gencode arch=compute_35,code=sm_35      \
			#	-gencode arch=compute_35,code=compute_35 \
			#	-DDEBUG -o $@ $^ -lopencv_core -lopencv_highgui
			nvcc imedgeGStr.cu -o imedgeGStr -lopencv_core -lopencv_highgui
					

imedgeG		:	imedgeG.cu
				nvcc imedgeG.cu -o imedgeG -lopencv_core -lopencv_highgui

debug		:	imedgeGStr_dbg

imedgeGStr_dbg	:	imedgeGStr.cu
				nvcc -g -G imedgeGStr.cu -o imedgeGStr_dbg

imedgeCL	:	imedgeCL.cpp
				g++ imedgeCL.cpp -o imedgeCL -lOpenCL -lopencv_core -lopencv_highgui
        				
clean		:
			rm -f imedgeG 

# REMINDER: type these module loads before trying to compile
#			module load gtk
#			module load gstreamer/0.10.36/b1
#			module load opencv
#			module load cuda
#			module load gcc/4.7.2
