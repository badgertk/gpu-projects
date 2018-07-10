#!/usr/bin/env python

import sqlite3
import argparse
import csv
import os
import time

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

with open("myRRs.sql", "r") as f:
	cur.executescript( f.read() )
con.commit()

script_time_end = time.time()
print("--- Table Population: %s seconds ---" % (script_time_end - script_time_start))
################## Print out results from database #############################
# featurelist_time_start = time.time()

# print 'features'
# print '========'
# for row in cur.execute('select * from FeatureList;'):
	# feature = row[0] #tuple to int
	# print feature
	
# print '========'
# for row in cur.execute('select count(feature) from FeatureList;'):
	# cnt = row[0] #tuple to int
	# print 'Number of features found: ' + str(cnt)
# print '========'
# featurelist_time_end = time.time()

# print("--- FeatureList Printout: %s seconds ---" % (featurelist_time_end - featurelist_time_start))
################## Finding the R to R (heart rate)##############################
RR_time_start = time.time()

for row in cur.execute('select count(feature) from myRRs;'):
	cnt = row[0]
	print 'Number of Rs found: ' + str(cnt)
lead1_data = cur.execute('select * from myRRs where lead=1;')

with open('lead1.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'lead', 'sample', 'feature'])
    writer.writerows(lead1_data)

for row in cur.execute('select count(feature) from myRRs where lead=1;'):
	cnt = row[0] #tuple to int
	print 'Number of Rs in lead 1: ' + str(cnt)
for row in cur.execute('select count(feature) from myRRs where lead=2;'):
	cnt = row[0] #tuple to int
	print 'Number of Rs in lead 2: ' + str(cnt)
for row in cur.execute('select count(feature) from myRRs where lead=3;'):
	cnt = row[0] #tuple to int
	print 'Number of Rs in lead 3: ' + str(cnt)
	
RR_time_end = time.time()
print("--- R Printout: %s seconds ---" % (RR_time_end - RR_time_start))
#################################### Done. #####################################
#TODO: 	db query can be outputed to .csv file
#		now stick that into gpu for processing

con.close()