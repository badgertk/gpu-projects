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
################################CPU Processing. ##############################

readRR_time_start = time.time()
sampling_rate=200.00
L1sample_no = []
L2sample_no = []
L3sample_no = []
L1time = []
L2time = []
L3time = []
with open('lead1RR.csv','rb') as csvfile:
	csvreader = csv.reader(csvfile)
	i = 0
	for row in csvreader:	
		#print ', '.join(row)
		if(i!=0):| i!=1):
			sample = int(row[2])
			L1sample_no.append(sample)
		i = i + 1

with open('lead2RR.csv','rb') as csvfile:
	csvreader = csv.reader(csvfile)
	i = 0
	for row in csvreader:	
		#print ', '.join(row)
		if(i!=0):| i!=1):
			sample = int(row[2])
			L2sample_no.append(sample)
		i = i + 1
with open('lead3RR.csv','rb') as csvfile:
	csvreader = csv.reader(csvfile)
	i = 0
	for row in csvreader:	
		#print ', '.join(row)
		if(i!=0):| i!=1):
			sample = int(row[2])
			L3sample_no.append(sample)
		i = i + 1
Diff_time_start = time.time()
L1sampleno = numpy.array(L1sample_no)
L2sampleno = numpy.array(L2sample_no)
L3sampleno = numpy.array(L3sample_no)
L1samplediff= numpy.diff(L1sampleno)
L2samplediff= numpy.diff(L2sampleno)
L3samplediff= numpy.diff(L3sampleno)
Diff_time_end = time.time()

HMS_time_start = time.time()
for row in L1sampleno:
	hour = int(row / (3600*sampling_rate))
	minute = int(row / (60*sampling_rate)) % (60)
	second = (row / sampling_rate)% (60)
	finalcsvdata = str(hour) + ":" + str(minute) + ":" + str(second)
	L1time.append(finalcsvdata)
for row in L2sampleno:
	hour = int(row / (3600*sampling_rate))
	minute = int(row / (60*sampling_rate)) % (60)
	second = (row / sampling_rate)% (60)
	finalcsvdata = str(hour) + ":" + str(minute) + ":" + str(second)
	L2time.append(finalcsvdata)
for row in L3sampleno:
	hour = int(row / (3600*sampling_rate))
	minute = int(row / (60*sampling_rate)) % (60)
	second = (row / sampling_rate)% (60)
	finalcsvdata = str(hour) + ":" + str(minute) + ":" + str(second)
	L3time.append(finalcsvdata)
HMS_time_end = time.time()
#print("--- SampleID to hour:minute:second : %s seconds ---" % (HMS_time_end - HMS_time_start))
HR_time_start = time.time()
L1lensamplediff = len(L1samplediff)
L2lensamplediff = len(L2samplediff)
L3lensamplediff = len(L3samplediff)
with open('lead1HR.csv', 'wb') as g:
	writer = csv.writer(g)
	writer.writerow(['time', 'HR per minute'])
	for x in range(0,L1lensamplediff):
		#print str(L1time[x])
		L1samplediff[x] = L1samplediff[x]*60/200
		writer.writerow([str(L1time[x])] + [str(L1samplediff[x])])
with open('lead2HR.csv', 'wb') as h:
	writer = csv.writer(h)
	writer.writerow(['time', 'HR per minute'])
	for x in range(0,L2lensamplediff):
		#print str(L2time[x])
		L2samplediff[x] = L2samplediff[x]*60/200
		writer.writerow([str(L2time[x])] + [str(L2samplediff[x])])
with open('lead3HR.csv', 'wb') as j:
	writer = csv.writer(j)
	writer.writerow(['time', 'HR per minute'])
	for x in range(0,L3lensamplediff):
		#print str(L3time[x])
		L3samplediff[x] = L3samplediff[x]*60/200
		writer.writerow([str(L3time[x])] + [str(L3samplediff[x])])



