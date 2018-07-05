.PHONY : clean, debug

all			: morphC morphG morphGBLAS morphGStrBLAS3 morphGfinal morphGShBLAS


morphC		:	morphC.c ImageStuff.c ImageStuff.h
			gcc morphC.c ImageStuff.c -o morphC -lpthread -Wformat -Wall -lm

morphG		:	morphologicalG.cu
				nvcc morphologicalG.cu -o morphG -lopencv_core -lopencv_highgui
				
morphGBLAS	:	morphGBLAS.cu
			nvcc -g -G -gencode arch=compute_35,code=sm_35      \
					-gencode arch=compute_35,code=compute_35 \
					-DDEBUG -o $@ $^ -lopencv_core -lopencv_highgui -lcublas			

morphGSh		:	morphGSh.cu
				nvcc morphGSh.cu -o morphGSh -lopencv_core -lopencv_highgui
				
morphGShBLAS		:	morphGShBLAS.cu
					nvcc -g -G -gencode arch=compute_35,code=sm_35      \
					-gencode arch=compute_35,code=compute_35 \
					-DDEBUG -o $@ $^ -lopencv_core -lopencv_highgui	-lcublas	
				
morphGStr		:	morphGStr.cu
				nvcc -g -G -gencode arch=compute_35,code=sm_35      \
					-gencode arch=compute_35,code=compute_35 \
					-DDEBUG -o $@ $^ -lopencv_core -lopencv_highgui		
					
morphGStrBLAS		:	morphGStrBLAS.cu
				nvcc -g -G -gencode arch=compute_35,code=sm_35      \
					-gencode arch=compute_35,code=compute_35 \
					-DDEBUG -o $@ $^ -lopencv_core -lopencv_highgui	-lcublas
					
morphGStrBLAS2		:	morphGStrBLAS_2.cu
				nvcc -g -G -gencode arch=compute_35,code=sm_35      \
					-gencode arch=compute_35,code=compute_35 \
					-DDEBUG -o $@ $^ -lopencv_core -lopencv_highgui	-lcublas	

morphGStrBLAS3		:	morphGStrBLAS_3.cu
					nvcc -g -G -gencode arch=compute_35,code=sm_35      \
					-gencode arch=compute_35,code=compute_35 \
					-DDEBUG -o $@ $^ -lopencv_core -lopencv_highgui	-lcublas			

morphGfinal		:	morphGfinal.cu
					nvcc -g -G -gencode arch=compute_35,code=sm_35      \
					-gencode arch=compute_35,code=compute_35 \
					-DDEBUG -o $@ $^ -lopencv_core -lopencv_highgui	-lcublas			
						

dilation		:	dilation.cu
				nvcc dilation.cu -o dilation -lopencv_core -lopencv_highgui

debug		:	dilation_dbg

dilation_dbg	:	dilation.cu
				nvcc -g -G dilation.cu -o dilation_dbg
				
clean		:
				rm -f  morphC morphG morphGBLAS morphGStrBLAS3 morphGfinal morphGShBLAS
				