#import numpy #need module load
from timer import Timer
import matplotlib.pyplot as plt #need module load
#import pycuda.autoinit
#import pycuda.driver as drv
import pycuda as cuda
from pycuda.compiler import SourceModule #issue here
import ishne
import sys
import imp
# number of samples: 0.06 - 0.1 * SAMPLING_RATE (QRS Time: 60-100ms)
num_samples = int(0.08 * 200)#ecg.sampling_rate)
# 0.08*200 samples/s = 16 samples for the QRS complex
# Measured correlated peak 7 samples after start of QRS (according to chase)
wavelet = numpy.zeros(num_samples).astype(numpy.float32) #initialize with zeros

# wavelet time span is fixed at 8 [0,8]
# wavelet resolution can be changed (see third argument) 
# I see no reason to snip off edge data points

#with Timer() as wavelet_time:
#0 - 8 with step of 0.125 is 64 points
#    db4_wavelet(cuda.Out(wavelet), numpy.float32(0), numpy.float32(0.125), grid=(1, 1), block=(int(ecg.sampling_rate), 1, 1))
print  "Daubechies 4 (db4) Generation: sec"#.format(wavelet_time.interval)

# Filter after processing
#with Timer() as Filter_time:
#    median_filter(cuda.In(correlated_wavelet), cuda.Out(filtered_result), numpy.int32(lead_size), grid=(1,1), block=(int(ecg.samplingrate),1,1))

print  "Filtered Processed Waveform with Median Filter: sec"#.format(Filter_time.interval)

# How many points to plot
chunk = 10000
span = 8
