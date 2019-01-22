.PHONY : clean, debug

gpuhypot	:	gpuhypot.cu
				nvcc gpuhypot.cu -o gpuhypot

debug		:	gpuhypot_dbg

gpuhypot_dbg	:	gpuhypot.cu
				nvcc -g -G gpuhypot.cu -o gpuhypot_dbg  # may also want -O0; default is -O3

clean		:
				rm -f gpuhypot gpuhypot_dbg
