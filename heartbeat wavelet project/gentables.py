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
	cur.executescript( f.read() )
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