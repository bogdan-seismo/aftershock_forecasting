import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import aftfore
import sys

#Run the forecasting routines (Omi method)
#for the Gorj data

if __name__ == '__main__':

    #Data (for Gorj, Marasesti and so on)
    #sequence = "Gorj"
    sequence = "Marasesti"
    if (sequence == "Gorj"):
       Data = pd.read_csv("gorj.dat", header=None, sep='\s+')
    elif(sequence == "Marasesti"):
       Data = pd.read_csv("marasesti2.dat", header=None, sep='\s+')
    else:
       print("Sequence is not defined")
       exit()

    Data.columns = ["T", "Mag"]
    
    #Periods
    t_start_hours = 0.0
    t_start = t_start_hours/24
    tinv_hours = 24.0
    tinv = tinv_hours/24
    itv_training1 =   [t_start, tinv]            #The time interval of training data [day]
    itv_forecast1   = [tinv, tinv+3/24]          #The range of the testing period [day]
    itv_forecast2   = [tinv, tinv+6/24]          #The range of the testing period [day]
    itv_forecast3   = [tinv, tinv+9/24]          #The range of the testing period [day]
    itv_forecast4   = [tinv, tinv+12/24]         #The range of the testing period [day]
    itv_forecast5   = [tinv, tinv+24/24]         #The range of the testing period [day]
    
    #File to plot
    file_name =  "forecast_"+sequence+"_learning_"+str(t_start_hours)+"h_"+str(tinv_hours)+"h.pdf"

    #File with parameters
    file_param = "forecast_"+sequence+"_learning_"+str(t_start_hours)+"h_"+str(tinv_hours)+"h.txt"

    print("Model and Forecast for %s sequence." % sequence)
    print("Learning period starts at %s hours after the mainshock." %str(t_start_hours))
    print("Learning period is of %s hours after the mainshock." % str(tinv_hours))
    
    #Training
    print('Training the model')
    model1 = aftfore.model()
    param1 = model1.mcmc(Data, itv_training1)
    #Forecast
    print('Calculating first forecast (for 3 hours)')
    prediction1 = model1.predict(itv_forecast1)
    df_pred1 = pd.DataFrame(prediction1["pred"])
    print('Calculating second forecast (for 6 hours)')
    prediction2 = model1.predict(itv_forecast2)
    df_pred2 = pd.DataFrame(prediction2["pred"])
    print('Calculating third forecast (for 9 hours)')
    prediction3 = model1.predict(itv_forecast3)
    df_pred3 = pd.DataFrame(prediction3["pred"])
    print('Calculating fourth forecast (for 12 hours)')
    prediction4 = model1.predict(itv_forecast4)
    df_pred4 = pd.DataFrame(prediction4["pred"])
    print('Calculating fifth forecast (for 24 hours)')
    prediction5 = model1.predict(itv_forecast5)
    df_pred5 = pd.DataFrame(prediction5["pred"])
    
    #Plot
    plt.figure(figsize=(8.27,11.69))
    plt.clf()
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',linewidth=1,titlesize=12)
    mpl.rc('pdf',fonttype=42)
    mpl.rc('xtick.major',width=1)
    mpl.rc('xtick.minor',width=1)
    mpl.rc('ytick.major',width=1)
    mpl.rc('ytick.minor',width=1)
    plt.semilogy(df_pred1["mag"]+0.05, df_pred1["probability_of_one_or_more_events"], 'r-', label="Forecast duration: 3 hours")
    plt.semilogy(df_pred2["mag"]+0.05, df_pred2["probability_of_one_or_more_events"], 'g-', label="Forecast duration: 6 hours")
    plt.semilogy(df_pred3["mag"]+0.05, df_pred3["probability_of_one_or_more_events"], 'b-', label="Forecast duration: 9 hours")
    plt.semilogy(df_pred4["mag"]+0.05, df_pred4["probability_of_one_or_more_events"], 'k-', label="Forecast duration: 12 hours")
    plt.semilogy(df_pred5["mag"]+0.05, df_pred5["probability_of_one_or_more_events"], 'y-', label="Forecast duration: 24 hours")
    plt.legend(loc="upper right")
    
    plt.ylim([0.001, 1])
    plt.xlim([2,9])
    plt.xlabel("Magnitude, M")
    plt.ylabel("Probability of events with magnitude >= M")
    plt.grid()
    plt.tight_layout(rect=[0.1,0.3,0.8,0.8])
    plt.savefig(file_name)  
    #plt.show()

    #Plot estimated parameters in a file
    orig_stdout = sys.stdout
    f = open(file_param, 'w')
    sys.stdout = f

    print("Model and Forecast for %s sequence." % sequence)
    print("Learning period starts at %s hours after the mainshock." %str(t_start_hours))
    print("Learning period is of %s hours after the mainshock." % str(tinv_hours))
    
    print('MAP estimate: ')
    [k,p,c,beta] = param1["para_mle"][["k", "p", "c", "beta"]]
    print( "k: %f" % k )
    print( "p: %f" % p )
    print( "c: %f" % c )
    print( "beta: %f" % beta )
    print(' ')

    #Uncertainties
    print('MCMC parameters: ')
    [k_std,p_std,c_std,beta_std] = param1["para_mcmc"][["k", "p", "c", "beta"]].std()
    print( "k_uncertainty: %f" % k_std )
    print( "p_uncertainty: %f" % p_std )
    print( "c_uncertainty: %f" % c_std )
    print( "beta_uncertainty: %f" % beta_std )

    sys.stdout = orig_stdout
    f.close()
