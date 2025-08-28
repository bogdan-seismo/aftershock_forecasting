#!/usr/bin/bash
#Run forecast for Gorj EQ sequence

#learn period: 0 - 3h, forecast: 3 - 6h
echo 'First period'
dir="gorj_forecast_0h-3h_3h-6h"
mkdir -p "${dir}"
python tgjiu2.py
mv *.pdf fore.txt param.pkl gorj_forecast*.dat ./"${dir}"
echo ' '

#learn period: 0 - 6h, forecast: 6 - 12h
echo 'Second period'
dir="gorj_forecast_0h-6h_6h-12h"
mkdir -p "${dir}"
python tgjiu3.py
mv *.pdf fore.txt param.pkl gorj_forecast*.dat ./"${dir}"
echo ' '

#learn period: 0 - 9h, forecast: 9 - 18h
echo 'Third period'
dir="gorj_forecast_0h-9h_9h-18h"
mkdir -p "${dir}"
python tgjiu4.py
mv *.pdf fore.txt param.pkl gorj_forecast*.dat ./"${dir}"
echo ' '

#learn period: 0 - 12h, forecast: 12 - 24h
echo 'Fourth period'
dir="gorj_forecast_0h-12h_12h-24h"
mkdir -p "${dir}"
python tgjiu5.py
mv *.pdf fore.txt param.pkl gorj_forecast*.dat ./"${dir}"
echo ' '

#learn period: 0 - 24h, forecast: 24 - 48h
echo 'Fifth period'
dir="gorj_forecast_0h-24h_24h-48h"
mkdir -p "${dir}"
python tgjiu6.py
mv *.pdf fore.txt param.pkl gorj_forecast*.dat ./"${dir}"
echo ' '
