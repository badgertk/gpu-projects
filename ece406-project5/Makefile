.PHONY : all, clean

all		:	imedgeG imedgeGStr imedgeGStrC

imedgeG		:	imedgeG.cu
				nvcc imedgeG.cu -o imedgeG -lopencv_core -lopencv_highgui
				
imedgeGStr	:	imedgeGStr.cu
			#nvcc -g -G -gencode arch=compute_35,code=sm_35      \
			#	-gencode arch=compute_35,code=compute_35 \
			#	-DDEBUG -o $@ $^ -lopencv_core -lopencv_highgui
			nvcc imedgeGStr.cu -o imedgeGStr -lopencv_core -lopencv_highgui
					

imedgeGStrC	:	imedgeGStrC.cu
				nvcc imedgeGStrC.cu -o imedgeGStrC -lopencv_core -lopencv_highgui

debug		:	imedgeGStr_dbg

imedgeGStr_dbg	:	imedgeGStr.cu
				nvcc -g -G imedgeGStr.cu -o imedgeGStr_dbg
				
clean		:
			rm -f imedgeG imedgeGStr imedgeGStr_dbg imedgeGStrC

			
# REMINDER: type these module loads before trying to compile
#			module load gtk
#			module load gstreamer/0.10.36/b1
#			module load opencv
#			module load cuda
#			module load gcc/4.7.2

# salloc -t 5 -A ece406 -p ece406 --gres=gpu:1 srun ./<program>