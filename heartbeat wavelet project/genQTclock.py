#!/usr/bin/env python

from ecgclock import QTClock

#!/usr/bin/env python

import sqlite3
import argparse
import csv
import os
import time
import numpy

import pycuda

########################### Make a nice circular plot ##########################
QTClock_time_start = time.time()
#my_clock = QTClock.QTClock('heartrate (bpm) for patient 9004')
my_HRfig = QTClock.ECGFigure(nrows=1, ncols=2, title='9004HR_CGPU')
my_QTfig = QTClock.ECGFigure(nrows=1, ncols=2, title='9004QT_GPU')
my_QRSfig  = QTClock.ECGFigure(nrows=1, ncols=2, title='9004QRS_GPU')
my_STfig = QTClock.ECGFigure(nrows=1, ncols=2, title='9004ST_GPU')
CPU_clock = QTClock.QTClock('CPU using python', parent_figure=my_HRfig, subplot=1)
GPU_clock =  QTClock.QTClock('GPU using CUDA', min_rad=0, max_rad=80, parent_figure=my_HRfig, subplot=2)
QT_clock =  QTClock.QTClock('GPU using CUDA', min_rad=40, max_rad=80, parent_figure=my_QTfig, subplot=1)
QRS_clock =  QTClock.QTClock('GPU using CUDA', min_rad=0, max_rad=30, parent_figure=my_QRSfig, subplot=1)
ST_clock =  QTClock.QTClock('GPU using CUDA', min_rad=0, max_rad=30, parent_figure=my_STfig, subplot=1)
# to change ranges: min_rad=100, max_rad=500
CPU_clock.add_recording('./lead1HR.csv', label='lead1')
CPU_clock.add_recording('./lead2HR.csv', label='lead2')
CPU_clock.add_recording('./lead3HR.csv', label='lead3')

GPU_clock.add_recording('./lead1GPUHR.csv', label='lead1')
GPU_clock.add_recording('./lead2GPUHR.csv', label='lead2')
GPU_clock.add_recording('./lead3GPUHR.csv', label='lead3')

QT_clock.add_recording('./lead1GPUQT.csv', label='lead1')
QT_clock.add_recording('./lead2GPUQT.csv', label='lead2')
QT_clock.add_recording('./lead3GPUQT.csv', label='lead3')

QRS_clock.add_recording('./lead1GPUQRS.csv', label='lead1')
QRS_clock.add_recording('./lead2GPUQRS.csv', label='lead2')
QRS_clock.add_recording('./lead3GPUQRS.csv', label='lead3')

ST_clock.add_recording('./lead1GPUST.csv', label='lead1')
ST_clock.add_recording('./lead2GPUST.csv', label='lead2')
ST_clock.add_recording('./lead3GPUST.csv', label='lead3')
    
CPU_clock.add_legend()
GPU_clock.add_legend()
QT_clock.add_legend()
QRS_clock.add_legend()
ST_clock.add_legend()

my_HRfig.save('9004HR_CGPU.png')
my_QTfig.save('9004QT_GPU.png')
my_QRSfig.save('9004QRS_GPU.png')
my_STfig.save('9004ST_GPU.png')

QTClock_time_end = time.time()
print("--- QTClock Generation: %s seconds ---" % (QTClock_time_end - QTClock_time_start))
print("----------------------------------------------------------------------------------")