import AftFore as aft
import math
import sys

#Run the forecasting routines (Omi method)
#for the Gorj data

if __name__ == '__main__':

    #Name of sequence
    seq_name = 'Marasesti'
    
    #Data (for Marasesti and so on)
    Data    = './AftFore/marasesti2.dat'    # The path of the date file
    
    #Periods
    t_learn = [0.0, 0.5]         # The range of the learning period [day]
    t_for   = [0.5, 1.0]         # The range of the testing period [day]

    #File with estimated parameters
    file_param = 'marasesti_forecast_0h-12h_12h-24h.dat'
    
    #Combines "Aft.Est" and "Aft.Fore" routines (some options are missing when using this function)
    #aft.EstFore(Data,t_learn,t_for)

    #Param estimation (from learning period)
    param = aft.Est(Data, t_learn)

    #Run the forecasting (it produces three figures, in two PDF files: param.pdf and fore.pdf; also the fore.txt and param.pkl file)
    aft.Fore(param, t_for, Data_test=Data)

    #Make cumulative number versus time plot (it will produce one PDF file: NT.pdf)
    mu = param['mu0'] + param['para']['mu1']
    mu_last = mu[-1]                               #Mmagnitude of detection (mu) at the end of the training window
    Mcomplete = mu[-1] + 2*param['para']['sigma']  #Magnitude of completeness (95%)
    mag_th = math.ceil(Mcomplete*10)/10            #The magnitude threshold mag_th for forecasting (round up of Mcomplete)
    print( "Mc: %.2f" % Mcomplete )
    print( "Mc_round: %.1f" % mag_th )
    t_for_end = t_for[1]                           #The end of the forecasting period [day]. The time interval [0, t_for_end]
    aft.NT_plot(Data, t_for_end, mag_th, param)

    #Plot estimated parameters in a file
    orig_stdout = sys.stdout
    f = open(file_param, 'w')
    sys.stdout = f

    #Print sequence name, learn and forecast periods
    print("Sequence name: %s" % seq_name)
    print(" ")
    print("Learn period:  %s" % str(t_learn))
    print("Forecast period: %s" % str(t_for))
    print(" ")
    
    #Print some magnitudes
    print( "mu_last: %.2f" % mu_last)
    print( "Mc_last: %.2f" % Mcomplete )
    print( "Mc_last_round: %.1f" % mag_th )
    print (" ")
    
    #Parameters
    [k,p,c,beta,mu1,sigma] = param["para"][["k", "p", "c", "beta","mu1","sigma"]]
    b_value = beta/(math.log(10))
    print("MAP estimate: ")
    print( "K: %.2e" % k )
    print( "p: %.2f" % p )
    print( "c: %.2e" % c )
    print( "beta: %.2f" % beta )
    print( "b-value: %.2f" % b_value)
    print( "mu1: %.2e" % mu1)
    print( "sigma: %.2e" % sigma )
    print(" ")

    #Uncertainties (standard deviations: to be used in the paper)
    [k_std,p_std,c_std,beta_std,mu1_std,sigma_std] = param["para_mcmc"][["k", "p", "c", "beta","mu1","sigma"]].std()
    print("MCMC parameters (standard deviations), to be used in the paper: ")
    print( "K_uncertainty: %.2e" % k_std )
    print( "p_uncertainty: %.2f" % p_std )
    print( "c_uncertainty: %.2e" % c_std )
    print( "beta_uncertainty: %.2f" % beta_std )
    print( "mu1_uncertainty: %.2e" % mu1_std )
    print( "sigma_uncertainty: %.2e" % sigma_std )
    print(" ")

    #Standard errors (determined with the Quasi Newton method)
    [k_ste,p_ste,c_ste,beta_ste,mu1_ste,sigma_ste] = param["ste"][["k", "p", "c", "beta","mu1","sigma"]]
    print("Standard errors: ")
    print( "K_ste: %.2e" % k_ste )
    print( "p_ste: %.2f" % p_ste )
    print( "c_ste: %.2e" % c_ste )
    print( "beta_ste: %.2f" % beta_ste )
    print( "mu1_ste: %.2e" % mu1_ste )
    print( "sigma_ste: %.2e" %sigma_ste )
    print(" ")
    
    sys.stdout = orig_stdout
    f.close()

