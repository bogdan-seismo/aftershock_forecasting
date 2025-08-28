#!/home/benescu/.pyenv/shims/python

#Reads an earthquake catalog file in the format below and outputs the time difference from a reference date and the magnitude.
#This information is needed to make the input catalog for the routines of Omi.

#Input file format (slighlty different format than in the transf_jma.py script):
#Long, Lat, Year, Month, Day, Magnitude, Depth, Hour, Minute

#Output file format
#Time (days) Magnitude

import math
import sys
import getopt
import os.path
import datetime as dt
from dateutil import relativedelta as rdelta

import matplotlib.pyplot as plt
import numpy as np

input_file = ''
output_file = ''

#Read input and output arguments
##############################################################
try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:h")
except getopt.GetoptError as e:
    print (str(e))
    print("Usage: %s -i input -o output" % sys.argv[0])
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print("Usage: %s -i input -o output" % sys.argv[0])
        sys.exit(2)
    elif opt == '-i':
        input_file = arg
    elif opt == '-o':
        output_file = arg

if input_file == "":
    print("The input file is not defined")
    sys.exit(2)

if output_file == "":
    print("The output file is not defined")
    sys.exit(2)

if not os.path.isfile(input_file):
    print("The input file does not exist!")
    sys.exit(2)
#############################################################

# Define type of data
#Format: long, lat, year, month, day, magnitude, depth, hour, min.
dtype1 = np.dtype([('year', 'i4'), ('month', 'i4'), ('day', 'i4'), ('hour', 'i4'), ('minute', 'i4')]) 
dtype2 = np.dtype([('longit', 'f8'), ('lat', 'f8'), ('magn', 'f8')])

# Reference time (for Vrancea-Marasesti sequence)
ref_year = 2014
ref_month = 11
ref_day = 22
ref_hour = 19
ref_min = 14

# Read data from input file (only the date and time)
with open(input_file, "r") as fi: 
     #timevalues = np.loadtxt(fi, dtype=dtype1, usecols=(2,3,4,7,8,9))
     curr_year, curr_month, curr_day, curr_hour, curr_min  = np.loadtxt(fi, dtype=dtype1, usecols=(2,3,4,7,8), unpack=True)

# Read data from input file (longitude, latitude, magnitude)
with open(input_file, "r") as fi:
    longit, lat, magn = np.loadtxt(fi, dtype=dtype2, usecols=(0,1,5), unpack=True)

# Find the time difference
ref_time = dt.datetime(ref_year,ref_month,ref_day,ref_hour,ref_min)
time_dif_list=[]
for i in range(len(curr_min)):
    time_vec = dt.datetime(curr_year[i],curr_month[i],curr_day[i],curr_hour[i],curr_min[i])
    #Changed to days
    time_dif = ((time_vec - ref_time).total_seconds())/(86400)
    # time_dif = (time_vec - ref_time).total_seconds()
    time_dif_list.append(time_dif)

# Save time difference and  magnitude in file
with open(output_file, "wb") as fo:
    #Changed the format for the time in years
    np.savetxt(fo, np.transpose([time_dif_list,magn]), fmt='%16.12f %5.1f')

    
