.PHONY : all, clean

all		:	RR

RR    : RR.cu
		nvcc RR.cu -o RR

#imrotateGS	:	imrotateGS.cu
#				nvcc imrotateGS.cu -o imrotateGS -lopencv_core -lopencv_highgui
				
clean		:
			rm -f RR

# REMINDER: type these module loads before trying to compile
#			module load gtk
#			module load gstreamer/0.10.36/b1
#			module load opencv
#			module load cuda
#			module load gcc/4.7.2