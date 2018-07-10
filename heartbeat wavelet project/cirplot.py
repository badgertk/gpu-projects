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

 ######################## Parse command line arguments: #########################

 parser = argparse.ArgumentParser(description='Extract all of the R deliniations')
 #parser.add_argument('csv_filename', help='csv file to convert')
 parser.add_argument('db_filename',  help='database filename', nargs='?')
 args = parser.parse_args()
 #if not args.db_filename:
 #    args.db_filename = os.path.splitext( args.db_filename )[0] + '.db'
 # We can now use args.csv_filename and args.db_filename when needed.

 ########################### Connect to the database: ###########################

 con = sqlite3.connect( args.db_filename )
 cur = con.cursor()

 ################## Populate the database using .sql file: ######################

 script_time_start = time.time()

 with open("mytables.sql", "r") as f:
	# cur.executescript( f.read() )
 con.commit()

 script_time_end = time.time()
 print("--- Table Population: %s seconds ---" % (script_time_end - script_time_start))

 ################## Generating CSV files ########################################
 RR_time_start = time.time()

 CSV_data = cur.execute('select * from myRRs where lead=1;')
 #cnt = cur.execute('select count(*) from myRRs where lead=1;')
 #print cnt
 with open('lead1RR.csv', 'wb') as a:
     writer = csv.writer(a)
     #writer.writerow(cnt)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)

 CSV_data = cur.execute('select * from myRRs where lead=2;')
 with open('lead2RR.csv', 'wb') as b:
     writer = csv.writer(b)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)

 CSV_data = cur.execute('select * from myRRs where lead=3;')
 with open('lead3RR.csv', 'wb') as c:
     writer = csv.writer(c)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)

 CSV_data = cur.execute('select * from myQT where lead=1;')
 with open('lead1QT.csv', 'wb') as a:
     writer = csv.writer(a)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)
 CSV_data = cur.execute('select * from myQT where lead=2;')
 with open('lead2QT.csv', 'wb') as a:
     writer = csv.writer(a)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)
 CSV_data = cur.execute('select * from myQT where lead=3;')
 with open('lead3QT.csv', 'wb') as a:
     writer = csv.writer(a)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)

 CSV_data = cur.execute('select * from myQRS where lead=1;')
 with open('lead1QRS.csv', 'wb') as a:
     writer = csv.writer(a)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)
 CSV_data = cur.execute('select * from myQRS where lead=2;')
 with open('lead2QRS.csv', 'wb') as a:
     writer = csv.writer(a)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)
 CSV_data = cur.execute('select * from myQRS where lead=3;')
 with open('lead3QRS.csv', 'wb') as a:
     writer = csv.writer(a)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)

 CSV_data = cur.execute('select * from myST where lead=1;')
 with open('lead1ST.csv', 'wb') as a:
     writer = csv.writer(a)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)
 CSV_data = cur.execute('select * from myST where lead=2;')
 with open('lead2ST.csv', 'wb') as a:
     writer = csv.writer(a)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)
 CSV_data = cur.execute('select * from myST where lead=3;')
 with open('lead3ST.csv', 'wb') as a:
     writer = csv.writer(a)
     writer.writerow(['id', 'lead', 'sample', 'feature'])
     writer.writerows(CSV_data)

	
 RR_time_end = time.time()
 print("--- R Printout: %s seconds ---" % (RR_time_end - RR_time_start))
 #################################### Done. #####################################

 ################################# CPU Processing. ##############################

 con.close()
# readRR_time_start = time.time()
# sampling_rate=200.00
# L1sample_no = []
# L2sample_no = []
# L3sample_no = []
# L1time = []
# L2time = []
# L3time = []
# with open('lead1RR.csv','rb') as csvfile:
	# csvreader = csv.reader(csvfile)
	# i = 0
	# for row in csvreader:	
		# #print ', '.join(row)
		# if(i!=0):# | i!=1):
			# sample = int(row[2])
			# L1sample_no.append(sample)
		# i = i + 1

# with open('lead2RR.csv','rb') as csvfile:
	# csvreader = csv.reader(csvfile)
	# i = 0
	# for row in csvreader:	
		# #print ', '.join(row)
		# if(i!=0):# | i!=1):
			# sample = int(row[2])
			# L2sample_no.append(sample)
		# i = i + 1
# with open('lead3RR.csv','rb') as csvfile:
	# csvreader = csv.reader(csvfile)
	# i = 0
	# for row in csvreader:	
		# #print ', '.join(row)
		# if(i!=0):# | i!=1):
			# sample = int(row[2])
			# L3sample_no.append(sample)
		# i = i + 1
# Diff_time_start = time.time()
# L1sampleno = numpy.array(L1sample_no)
# L2sampleno = numpy.array(L2sample_no)
# L3sampleno = numpy.array(L3sample_no)
# L1samplediff= numpy.diff(L1sampleno)
# L2samplediff= numpy.diff(L2sampleno)
# L3samplediff= numpy.diff(L3sampleno)
# Diff_time_end = time.time()

 HMS_time_start = time.time()
 for row in L1sampleno:
	# hour = int(row / (3600*sampling_rate))
	# minute = int(row / (60*sampling_rate)) % (60)
	# second = (row / sampling_rate)% (60)
	# finalcsvdata = str(hour) + ":" + str(minute) + ":" + str(second)
	# L1time.append(finalcsvdata)
 for row in L2sampleno:
	# hour = int(row / (3600*sampling_rate))
	# minute = int(row / (60*sampling_rate)) % (60)
	# second = (row / sampling_rate)% (60)
	# finalcsvdata = str(hour) + ":" + str(minute) + ":" + str(second)
	# L2time.append(finalcsvdata)
 for row in L3sampleno:
	# hour = int(row / (3600*sampling_rate))
	# minute = int(row / (60*sampling_rate)) % (60)
	# second = (row / sampling_rate)% (60)
	# finalcsvdata = str(hour) + ":" + str(minute) + ":" + str(second)
	# L3time.append(finalcsvdata)
 HMS_time_end = time.time()
 #print("--- SampleID to hour:minute:second : %s seconds ---" % (HMS_time_end - HMS_time_start))
# HR_time_start = time.time()
# L1lensamplediff = len(L1samplediff)
# L2lensamplediff = len(L2samplediff)
# L3lensamplediff = len(L3samplediff)
# with open('lead1HR.csv', 'wb') as g:
	# writer = csv.writer(g)
	# writer.writerow(['time', 'HR per minute'])
	# for x in range(0,L1lensamplediff):
		# #print str(L1time[x])
		# L1samplediff[x] = L1samplediff[x]*60/200
		# writer.writerow([str(L1time[x])] + [str(L1samplediff[x])])
# with open('lead2HR.csv', 'wb') as h:
	# writer = csv.writer(h)
	# writer.writerow(['time', 'HR per minute'])
	# for x in range(0,L2lensamplediff):
		# #print str(L2time[x])
		# L2samplediff[x] = L2samplediff[x]*60/200
		# writer.writerow([str(L2time[x])] + [str(L2samplediff[x])])
# with open('lead3HR.csv', 'wb') as j:
	# writer = csv.writer(j)
	# writer.writerow(['time', 'HR per minute'])
	# for x in range(0,L3lensamplediff):
		# #print str(L3time[x])
		# L3samplediff[x] = L3samplediff[x]*60/200
		# writer.writerow([str(L3time[x])] + [str(L3samplediff[x])])

 #print("Lead 3 HR average = %s" % numpy.average(L3samplediff))

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
