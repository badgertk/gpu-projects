#!/usr/bin/env python

#### Imports ####

import csv
import numpy as np
from dateutil import parser
import datetime
import fnmatch
import os
import argparse

#### Functions ####

def sample_to_clock(sample_num, start_time, sr):
    """start_time is a string like '11:06'.  sr in Hz."""
    start_time = parser.parse(start_time)
    offset_sec = 1.0*sample_num/sr
    time = start_time + datetime.timedelta(seconds=offset_sec)
    return time.isoformat()

#### Parse input args: ####

argparser = argparse.ArgumentParser(
    description="Convert Gilead data to csv format for QTClock library to use."
)
argparser.add_argument("-s", "--sr",
                       help="sample rate in Hz (default=1000)",
                       type=int, default=1000)
argparser.add_argument("directory",
                       help="directory with files to parse (containing one recording)")
args = argparser.parse_args()

#### Variables ####

sr = args.sr
conv_dir = args.directory

tsv_files = []
start_time_file = ''
for fn in os.listdir(conv_dir):
    if fnmatch.fnmatch(fn, 'IntervalData.xls'):
        start_time_file = os.path.join(conv_dir,fn)
    if fnmatch.fnmatch(fn, 'pat*QT*.xls'):
        # TODO: use extended glob or regex or something better than '*' to match?
        tsv_files += [ os.path.join(conv_dir,fn) ]
assert ((len(tsv_files) != 0) and (start_time_file != ''))
tsv_files.sort()
tsv_base_filename = tsv_files[0][:tsv_files[0].rfind('QT')+2]

#### Find start time ####

with open(start_time_file, 'rtU') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t')
    mycsv = list(csv_reader)
    date = mycsv[1][1].strip()
    time = mycsv[1][2].strip()
time = ':'.join( [ num.zfill(2) for num in time.split(':') ] )

start_time = date + 'T' + time

#### Load {sample, QT (ms), RR (ms)} from tsv file(s) ####

cols_wanted = [0,8,3]
col_fmt=[int,float,float]
results = [ [] for i in range(len( cols_wanted )) ]
for tsv_filename in tsv_files:
    with open(tsv_filename, 'rtU') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if (i==0):
                continue  # we know these files have headers
            new_results = ['' for _ in range(len( cols_wanted ))]
            try:
                for j, col in enumerate(cols_wanted):
                    new_results[j] = col_fmt[j]( row[col] )
            except ValueError:
                continue  # skip bad values; usually caused by empty cells
            for j, res in enumerate(new_results):
                results[j].append( res )
sample_num, qt, rr = tuple(results)

#### Convert results to {time, QTc (sec)} ####

qt = [val / 1000.0 for val in qt]
rr = [val / 1000.0 for val in rr]
values = list(np.divide( qt, np.power(rr, 1.0/3) ))
times = [sample_to_clock(t, start_time, sr) for t in sample_num]

#### Save to QTClock-compatible csv ####

output_csv_file = tsv_base_filename + '.csv'

with open(output_csv_file, 'w') as csvfile:  # existing file will be overwritten
    spamwriter = csv.writer(csvfile, delimiter=',')
    for i in range( len(times) ):
        this_row = [times[i]] + [values[i]]
        spamwriter.writerow(this_row)

#### Done.  You can now make a clock from output_csv_file. ####
