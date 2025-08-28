Codes by Takahiro Omi and adapted for the Romanian seismicity by Bogdan Enescu.

The codes produce forecasting results (e.g., Figure 7 of Ghita et al., 2025).

Please also see the readme.ipynb for the usage detail (it is a Jupyter notebook file).

The file "pred_eq_prob.py" is used to produce forecasts for the Marasesti and Gorj (Tg. Jiu) sequences in Romania.
The data is in "gorj.dat" and "marasesti2.dat".
The format is relative time from the mainshock (in days) and magnitude.

If you use these codes, please cite the following papers:
- Omi, T., Ogata, Y., Shiomi, K., Enescu, B., Sawazaki, K., and K. Aihara, Automatic aftershock forecasting: a test using real-time seismicity data in Japan. 
Bull Seismol Soc Am 106:2450–2458, 2016. https://doi.org/10.1785/0120160100
- Omi, T., Ogata, Y., Shiomi, K., Enescu, B., Sawazaki, K., and K. Aihara, Implementation of a Real‐Time System for Automatic Aftershock Forecasting in Japan. 
Seismological Research Letters, 90(1): 242–250, 2019. doi: https://doi.org/10.1785/0220180213
- Ghita, C., Enescu, B., Marinus, A., Moldovan, I.-A., Ionescu, C., Constantinescu, E.G. and L. An, Aftershock analysis and forecasting for the crustal seismicity in Romania. 
Earth Planets Space 77, 48, (2025). https://doi.org/10.1186/s40623-025-02174-0

Important note: the code uses Python 3.11.
You can use pyenv to manage various versions of Python on the same machine.
https://www.dwarmstrong.org/pyenv/
I think different Conda environments would also work.

