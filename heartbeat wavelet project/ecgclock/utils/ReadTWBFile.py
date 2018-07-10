#!/usr/bin/python

"""Read a COMPAS TWB file."""

########## Imports: ##########

import sys, inspect
import os
import numpy as np
import datetime
import csv

########## Functions: ##########

def ReadTWBFile(fileName, max_beats=500000):
    """Read a COMPAS TWB file and return a 2D array of data for each beat.  The
    output will have the following columns, with one row for every beat:
      day / time (2 columns)
      pTWBData (numColumn columns)
      beatStability (1 column)
      Beat_Anno (1 column)
      LTangent (numLead columns)
      RTangent (numLead columns)
      TLength  (numLead columns)
      TQ       (numLead columns)
    The pTWBData columns are:
      RR, Ampl_[all leads], QTApex_[all leads], RTApex_[all leads],
      QTOffset_[all leads], RTOffset_[all leads],
      QTA25_[all leads], RTA25_[all leads],
      QTA50_[all leads], RTA50_[all leads],
      QTA95_[all leads], RTA95_[all leads],
      QTA97_[all leads], RTA97_[all leads]
    QTc could be computed from RR and QTOffset.  Note that -9 in any field
    indicates a "non-measurable" value.  Also, sinus beats will all have
    Beat_Anno=0.

    Keyword arguments:
    fileName -- the file to read
    max_beats -- if the file claims to have annotations for more than this number of beats, we won't process it
    """

    lFileLength = os.path.getsize(fileName)
    br = open(fileName, 'rb')

    # get number of beats (line of data):
    br.seek(lFileLength-4, os.SEEK_SET)
    numBeat = np.fromfile(br, dtype=np.int32, count=1)[0]

    if ( (numBeat<0) or (numBeat>max_beats) ):
        br.close()
        sys.exit("Got bad value (" + str(numBeat) + ") for number of beats in " + fileName + "; exiting.")

    br.seek(0, os.SEEK_SET)
    
    # read beat to beat header.  Beat-to-beat .twb files should always start with 1.
    value = np.fromfile(br, dtype=np.int32, count=1)[0]
    assert (value == 1), "Can't handle this type of input file!"
    fileNameLen = np.fromfile(br, dtype=np.int32, count=1)[0]
    charFN = np.fromfile(br, dtype=np.uint8, count=fileNameLen)
    charFN = ''.join(chr(i) for i in charFN)
    value = np.fromfile(br, dtype=np.int32, count=5)
    numLead = np.fromfile(br, dtype=np.int32, count=1)[0]

    # prepare empty arrays:
    numColumn = 1+numLead + 6*2*numLead
    pTWBData = [[0 for col in range(numColumn)] for row in range(numBeat)]
    pTime = [0 for i in range(numBeat)]
    pBeatAnnotation = [0 for i in range(numBeat)]
    beatStability = [0 for i in range(numBeat)]
    LTangent = [[0 for col in range(numLead)] for row in range(numBeat)]
    RTangent = [[0 for col in range(numLead)] for row in range(numBeat)]
    TLength  = [[0 for col in range(numLead)] for row in range(numBeat)]
    TQ       = [[0 for col in range(numLead)] for row in range(numBeat)]
    
    # read data:
    for i in range(numBeat):
        pTime[i] =           np.fromfile(br, dtype=np.int32,   count=1        )[0]
        sampleNum =          np.fromfile(br, dtype=np.int32,   count=1        )[0]  # not used
        pTWBData[i]  = list( np.fromfile(br, dtype=np.float32, count=numColumn) )
        beatStability[i]   = np.fromfile(br, dtype=np.uint8,   count=1        )[0]
        pBeatAnnotation[i] = np.fromfile(br, dtype=np.uint8,   count=1        )[0]
        TwaveAnnotation    = np.fromfile(br, dtype=np.uint8,   count=numLead  )     # not used
        LTangent[i]  = list( np.fromfile(br, dtype=np.float32, count=numLead  ) )
        RTangent[i]  = list( np.fromfile(br, dtype=np.float32, count=numLead  ) )
        TLength[i]   = list( np.fromfile(br, dtype=np.float32, count=numLead  ) )
        TQ[i]        = list( np.fromfile(br, dtype=np.float32, count=numLead  ) )
        SNR                = np.fromfile(br, dtype=np.float32, count=numLead  )     # not used

    br.close()

    # make single 2D array with all results.  posix times will be converted to
    # 'day' and 'time' strings.
    output_array = [ [0 for col in range(2 + numColumn + 1 + 1 + 4*numLead)]
                        for row in range(numBeat) ]  # no row for header line
    for row in range(numBeat):
        dt = datetime.datetime.fromtimestamp(pTime[row])   # use this and specify known timezone,
        # dt = datetime.datetime.utcfromtimestamp(pTime[row])  # or assume local time was stored

        # TWB time format is ridiculous.  The times appear to be adjusted to GMT
        # based on the time zone of the annotator, rather than the patient.  For
        # example, if the TWB is created in EST (-0500), you must use EST in
        # fromtimestamp() when you read them, not your local time zone or GMT.

        day = dt.strftime("%a %b %d %Y")  # TODO: verify format
        time = dt.strftime("%H:%M:%S")  # TODO: verify format
        output_array[row] = [day] + [time] + pTWBData[row] +\
                            [beatStability[row]] + [pBeatAnnotation[row]] + LTangent[row] +\
                            RTangent[row] + TLength[row] + TQ[row]

    return output_array

########## Main code: ##########

if __name__ == "__main__":

    if (len(sys.argv) == 2):
        # No output filename specified, we'll make it input_filename.csv.
        per = sys.argv[1].rfind('.')
        if (per != -1):
            outFile = sys.argv[1][:per] + '.csv'
        else:
            outFile = sys.argv[1] + '.csv'
        # TODO: make sure output filename isn't same as input filename
    elif (len(sys.argv) == 3):
        outFile = str( sys.argv[2] )
    else:
        sys.exit('Incorrect number of arguments.  Format is:\n ' + 
                 inspect.getfile(inspect.currentframe()) + 
                 ' <input TWB filename> [output CSV filename, default=input_filename.csv]')

    #print "output=" + outFile  # debugging

    # Skip existing files:
    if os.path.isfile(outFile):
        sys.exit("Output file (" + outFile + ") already exists; exiting.")

    fileName = str( sys.argv[1] )
    results = ReadTWBFile(fileName) 

    # save results to CSV:
    with open(outFile, 'w') as csvfile:  # existing file will be overwritten
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in results:
            spamwriter.writerow(row)
    # TODO: maybe add a header row.  need to know what the headers should be
    # based on number of leads, though.  maybe we can just number them 1-N
    # rather than naming them?  For the 8-lead recordings we plan to look at,
    # the order is always I, II, V1-V6.

################################
