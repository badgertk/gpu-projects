#!/usr/bin/env python

import pycuda.autoinit
import pycuda.driver as cuda
import numpy
import scipy.signal
import matplotlib.pyplot as plt
import ishne
from pycuda.compiler import SourceModule
import sys
from timer import Timer
from table import Table
import computedb

with open("wavelet2.cu") as wavelet_file:
    mod = SourceModule(wavelet_file.read())

mexican_hat = mod.get_function("mexican_hat")
cross_correlate_with_wavelet = mod.get_function("cross_correlate_with_wavelet")
threshold = mod.get_function("threshold")
edge_detect = mod.get_function("edge_detect")
get_rr = mod.get_function("get_rr")
filter = mod.get_function("int_3_median_filter")
median = mod.get_function("int_3_median_reduction")
get_rr_2 = mod.get_function("get_rr_2")
index_of_peak = mod.get_function("index_of_peak")
get_sample2 = mod.get_function("get_sample2")
get_rr_3 = mod.get_function("get_rr_3")
merge_leads = mod.get_function("merge_leads")

def median_filter(out_array, in_ary, grid, block):
    padded = numpy.pad(in_ary, (1, 1), mode="edge")
    filter(cuda.Out(out_array), cuda.In(padded), grid=grid, block=block)
    return out_array

def preprocess_lead(lead, lead_size, d_wavelet, wavelet_len, threshold_value):

    # Kernel Parameters
    threads_per_block = 200
    num_blocks = lead_size / threads_per_block


    # transfer lead to device
    d_lead = cuda.to_device(lead)

    # correlate lead with wavelet
    correlated = cuda.mem_alloc(lead.nbytes)
    cross_correlate_with_wavelet(correlated, d_lead, d_wavelet,
                                 numpy.int32(lead_size),
                                 numpy.int32(wavelet_len),
                                 grid=(num_blocks, 1),
                                 block=(threads_per_block, 1, 1))

    # threshold correlated lead
    thresholded_signal = cuda.mem_alloc(lead.nbytes)
    threshold(thresholded_signal, correlated,
              numpy.float32(threshold_value),
              grid=(num_blocks, 1), block=(threads_per_block, 1, 1))

    # cleanup
    correlated.free()
    d_lead.free()

    # return result
    return thresholded_signal

def preprocess(lead1, lead2, lead3, wavelet, threshold_value):
    
    lead_size = len(lead1)
    d_wavelet = cuda.to_device(wavelet)
    wavelet_len = len(wavelet)
    lead_gen = (preprocess_lead(lead.astype(numpy.float32), lead_size, d_wavelet, wavelet_len, threshold_value)
                for lead in (lead1, lead2, lead3))
    (d_tlead1, d_tlead2, d_tlead3) = tuple(lead_gen)
    # synchronize
    (offset1, offset2, offset3, lead_len) = synchronize(d_tlead1, d_tlead2, d_tlead3, lead_size)
    # merge
    d_merged_lead, lead_len = merge(d_tlead1, offset1, d_tlead2, offset2, d_tlead3, offset3, lead_len)
    # cleanup
    for lead in (d_tlead1, d_tlead2, d_tlead3):
        lead.free()
    d_wavelet.free()
    # return results
    return (d_merged_lead, lead_len)

def synchronize(d_tlead1, d_tlead2, d_tlead3, length):
    # Number of points to use to synchronize
    chunk = ecg.sampling_rate * 2
    template = numpy.zeros(chunk).astype(numpy.int32)
    tlead1 = cuda.from_device_like(d_tlead1, template)
    tlead2 = cuda.from_device_like(d_tlead2, template)
    tlead3 = cuda.from_device_like(d_tlead3, template)
    start1 = numpy.argmax(tlead1)
    start2 = numpy.argmax(tlead2)
    start3 = numpy.argmax(tlead3)
    minstart = min(start1, start2, start3)
    offset1 = start1 - minstart
    offset2 = start2 - minstart
    offset3 = start3 - minstart
    new_length = length - minstart
    return (offset1, offset2, offset3, new_length)

def merge(d_slead1, offset1, d_slead2, offset2, d_slead3, offset3, length):
    # Kernel Parameters
    threads_per_block = 200
    num_blocks = length / threads_per_block

    d_merged_lead = cuda.mem_alloc(4 * num_blocks * threads_per_block)
    merge_leads(d_merged_lead,
                d_slead1, numpy.int32(offset1),
                d_slead2, numpy.int32(offset2),
                d_slead3, numpy.int32(offset3),
                grid=(num_blocks, 1), block=(threads_per_block, 1, 1))
    return d_merged_lead, num_blocks * threads_per_block

def get_heartbeat(d_lead, length):
    # Kernel Parameters
    threads_per_block = 200
    num_blocks = length / threads_per_block

    # Allocate output
    full_rr_signal = numpy.zeros(length).astype(numpy.int32)

    # Get RR
    window_size = 8
    edge_signal = cuda.mem_alloc(4 * length)
    edge_detect(edge_signal, d_lead, grid=(num_blocks, 1), block=(threads_per_block, 1, 1))
    indecies = numpy.zeros(length / 64).astype(numpy.int32)
    masks = cuda.to_device(numpy.zeros(length / 64).astype(numpy.int32))
    d_index = cuda.to_device(indecies)
    index_of_peak(d_index, masks, edge_signal, grid=(num_blocks, 1), block=(threads_per_block, 1, 1))
    get_rr_2(cuda.InOut(full_rr_signal), d_index, masks, numpy.int32(window_size), numpy.int32(ecg.sampling_rate), numpy.int32(len(indecies)), grid=(num_blocks / 64, 1), block=(threads_per_block, 1, 1))
    rr_signal = full_rr_signal[full_rr_signal != 0]

    # Filter
    smoothed_rr_signal = rr_signal[rr_signal < 120]
    smoothed_rr_signal = smoothed_rr_signal[smoothed_rr_signal > 10]
    smoothed_rr_signal2 = numpy.copy(smoothed_rr_signal)
    for i in range(3):
        if len(smoothed_rr_signal2) > 2187 * 3:
            median(cuda.Out(smoothed_rr_signal2), cuda.In(numpy.copy(smoothed_rr_signal2)), grid=(len(smoothed_rr_signal2) / 2187, 1), block=(729, 1, 1))
        elif 1024 < len(smoothed_rr_signal2) <= 2187 * 3:
            median(cuda.Out(smoothed_rr_signal2), cuda.In(numpy.copy(smoothed_rr_signal2)), grid=(len(smoothed_rr_signal2) / 729, 1), block=(81, 1, 1))
        else:
            median(cuda.Out(smoothed_rr_signal2), cuda.In(numpy.copy(smoothed_rr_signal2)), grid=(1, 1), block=(len(smoothed_rr_signal2), 1, 1))
        smoothed_rr_signal2 = smoothed_rr_signal2[:len(smoothed_rr_signal2)/3]
        if len(smoothed_rr_signal2) > 2187 * 3:
            median_filter(smoothed_rr_signal2, numpy.copy(smoothed_rr_signal2), grid=(len(smoothed_rr_signal2) / 2187, 1), block=(729, 1, 1))
        elif 1024 < len(smoothed_rr_signal2) <= 2187 * 3:
            median_filter(smoothed_rr_signal2, numpy.copy(smoothed_rr_signal2), grid=(len(smoothed_rr_signal2) / 729, 1), block=(81, 1, 1))
        else:
            median_filter(smoothed_rr_signal2, numpy.copy(smoothed_rr_signal2), grid=(9, 1), block=(len(smoothed_rr_signal2) / 9, 1, 1))
    smoothed_rr_signal2 = scipy.signal.medfilt(smoothed_rr_signal2, (31,))

    # Cleanup
    d_lead.free()
    edge_signal.free()
    d_index.free()
    masks.free()

    return smoothed_rr_signal2

def main():
    # Read the ISHNE file
    global ecg
    ecg = ishne.ISHNE(sys.argv[1])
    ecg.read()

    # number of samples: 0.06 - 0.1 * SAMPLING_RATE (QRS Time: 60-100ms)
    num_samples = int(0.08 * ecg.sampling_rate)
    # The math suggests 16 samples is the width of the QRS complex
    # Measuring the QRS complex for 9004 gives 16 samples
    # Measured correlated peak 7 samples after start of QRS
    h_hat = numpy.zeros(num_samples).astype(numpy.float32)
    # Mexican hats seem to hold a nonzero value between -4 and 4 w/ sigma=1
    sigma = 1.0
    maxval = 4 * sigma
    minval = -maxval

    hat = numpy.zeros(16).astype(numpy.float32)
    mexican_hat(cuda.Out(hat),
                numpy.float32(sigma),
                numpy.float32(minval),
                numpy.float32((maxval - minval)/num_samples),
                grid=(1, 1), block=(int(ecg.sampling_rate), 1, 1))

    with Timer() as pre_time:
        d_mlead, length = preprocess(ecg.leads[0], ecg.leads[1], ecg.leads[2], hat, 1.0)
    with Timer() as time:
        y = get_heartbeat(d_mlead, length)
    print pre_time.interval, time.interval, pre_time.interval + time.interval
    x = numpy.linspace(0, 23, num=len(y))
    plt.figure(2)
    plt.plot(x, y, 'b')
    plt.show()

if __name__ == '__main__':
    main()
