AftFore is a python package to generate aftershock forecasts based on a method proposed in "Omi et al.(2013)". 

Anyone can freely use this code without permission for non-commercial purposes, but please appropriately cite "Omi et al.(2013)" if you present results obtained by using AftFore.

- Omi, T., Ogata, Y., Hirata, Y., and K. Aihara, Forecasting large aftershocks within one day after the main shock, 
Scientific Reports 3, 2218, 2013. http://dx.doi.org/10.1038/srep02218

Documentation: https://omitakahiro.github.io/AftFore/

Developed by:"Takahiro Omi" (https://sites.google.com/view/omitakahiro/)

---------------

The package has been modified for analysing events in Romania, in Marasesti and Gorj (Tg. Jiu) regions.
The Python routines mara2.py, mara3.py, mara4.py, mara5.py and mara6.py are for the Marasesti sequence, for different learning and forecasting periods.

The package has been modified for analysing events in Romania, in Marasesti and Gorj regions.
The Python routines tgjiu2.py, tgjiu3.py, tgjiu4.py, tgjiu5.py and tgjiu6.py are for the Marasesti sequence, for different learning and forecasting periods.

The Bash routines "run_forecast_mara.sh" and "run_forecast_gorj.sh" are for the Marasesti and Gorj sequences, respectively.
They are the responsible for running the AftFore routines for different learning and forecasting windows.

The data used is in the "/AftFore" sub-directory (e.g., "marasesti2.dat", "gorj.dat", etc...).
The format of the data is relative time from the mainshock (in days) and magnitude.
The first line is for the mainshock.

If you use these routines, please cite besides Omi et al. (2013), the following papers: Omi et al. (2019) and Ghita et al. (2025).
- Omi, T., Ogata, Y., Shiomi, K., Enescu, B., Sawazaki, K., and K. Aihara, Implementation of a Real‐Time System for Automatic Aftershock Forecasting in Japan. 
Seismological Research Letters, 90(1): 242–250, 2019. doi: https://doi.org/10.1785/0220180213
- Ghita, C., Enescu, B., Marinus, A., Moldovan, I.-A., Ionescu, C., Constantinescu, E.G. and L. An, Aftershock analysis and forecasting for the crustal seismicity in Romania. 
Earth Planets Space 77, 48, (2025). https://doi.org/10.1186/s40623-025-02174-0

Important: These routines work with Python 2.7. You can use pyenv to manage various versions of Python on the same machine.
https://www.dwarmstrong.org/pyenv/
I think different Conda environments would also work.








