#!/usr/bin/env python

import sqlite3
import argparse
import csv
import os

######################## Parse command line arguments: #########################

parser = argparse.ArgumentParser(description='Convert a csv file of ECG annotations into a sqlite database.')
parser.add_argument('csv_filename', help='csv file to convert')
parser.add_argument('db_filename',  help='output database filename', nargs='?')
args = parser.parse_args()
if not args.db_filename:
    args.db_filename = os.path.splitext( args.csv_filename )[0] + '.db'
# We can now use args.csv_filename and args.db_filename when needed.

########################### Connect to the database: ###########################

con = sqlite3.connect( args.db_filename )
cur = con.cursor()

################## Initialize the database using schema.sql: ###################

with open("schema.sql", "r") as f:
    cur.executescript( f.read() )
con.commit()
    
################ Populate the "anns" table using the csv file: #################

with open(args.csv_filename, "r") as f:
    csvreader = csv.reader(f)
    for i, row in enumerate(csvreader):
        if (i == 0):
            continue  # skip the first row, which just tells us sample rate
        cur.execute('insert into anns (lead, sample, feature) values (?, ?, ?)', row)
con.commit()

#################################### Done. #####################################

con.close()

################################################################################
