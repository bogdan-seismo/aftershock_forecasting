#!/home/benescu/.pyenv/shims/python

#Reads an earthquake catalog file in the format below and outputs the time difference from a reference date and the magnitude.
#This information is needed to make the input catalog for the routines of Omi.

#Input file format:
#Long, Lat, Year, Month, Day, Magnitude, Depth, Hour, Minute, Second (can be with decimals)

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
#Type of input_file  = '../jma_catalog/jmacat_20010101_20161124.dat'
#Format: long, lat, year, month, day, magnitude, depth, hour, min., sec
dtype1 = np.dtype([('year', 'i4'), ('month', 'i4'), ('day', 'i4'), ('hour', 'i4'), ('minute', 'i4'), ('sec', 'f8')]) 
dtype2 = np.dtype([('longit', 'f8'), ('lat', 'f8'), ('magn', 'f8')])

# Reference time (for Tg. Jiu sequence)
ref_year = 2023
ref_month = 2
ref_day = 14
ref_hour = 13
ref_min = 16
ref_sec = 52

# Read data from input file (only the date and time)
with open(input_file, "r") as fi: 
     #timevalues = np.loadtxt(fi, dtype=dtype1, usecols=(2,3,4,7,8,9))
     curr_year, curr_month, curr_day, curr_hour, curr_min, curr_sec = np.loadtxt(fi, dtype=dtype1, usecols=(2,3,4,7,8,9), unpack=True)

#Find seconds (integer) and microseconds
curr_sec_int = np.array([])
curr_microsec = np.array([])
for i in range(len(curr_sec)):
    curr_sec_intl = math.floor(curr_sec[i])
    curr_microsecl = int((curr_sec[i] - curr_sec_intl)*1000000)
    curr_sec_int = np.append(curr_sec_int, curr_sec_intl)
    curr_microsec = np.append(curr_microsec, curr_microsecl)

ref_sec_int = math.floor(ref_sec)
ref_microsec = int((ref_sec - ref_sec_int)*1000000)

# Read data from input file (longitude, latitude, magnitude)
with open(input_file, "r") as fi:
    longit, lat, magn = np.loadtxt(fi, dtype=dtype2, usecols=(0,1,5), unpack=True)

# Find the time difference
ref_time = dt.datetime(ref_year,ref_month,ref_day,ref_hour,ref_min,int(ref_sec_int),int(ref_microsec))
time_dif_list=[]
for i in range(len(curr_sec)):
    time_vec = dt.datetime(curr_year[i],curr_month[i],curr_day[i],curr_hour[i],curr_min[i],int(curr_sec_int[i]),int(curr_microsec[i]))
    #Changed to days
    time_dif = ((time_vec - ref_time).total_seconds())/(86400)
    # time_dif = (time_vec - ref_time).total_seconds()
    time_dif_list.append(time_dif)

# Save time difference and  magnitude in file
with open(output_file, "wb") as fo:
    #Changed the format for the time in years
    np.savetxt(fo, np.transpose([time_dif_list,magn]), fmt='%16.12f %5.1f')
    #Old format for the time in seconds
    #np.savetxt(fo, np.transpose([time_dif_list,magn,lat,longit]), fmt='%14.2f %5.1f %8.4f %9.4f')
