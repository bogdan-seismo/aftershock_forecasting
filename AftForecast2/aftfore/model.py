import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.special import gammaln
from scipy.stats import norm,poisson
from scipy import sparse
import scipy.sparse.linalg as spla

import sys,time,copy,pickle,datetime,subprocess,itertools,json,os,traceback,logging

try:
    from .StatTool import Quasi_Newton,MCMC
except:
    from StatTool import Quasi_Newton,MCMC

"""
model description:
lambda(t,M) = k*(t+c)**(-p) * beta*exp(-beta(M-mag_ref)) * Phi(M|mu(t),sigma)
"""
#######################################################################
## generic values
#######################################################################
def prior_generic():
    prior =[]
    prior.append({'name':'beta',  'index':0, 'type':'n',  'mu':0.85*np.log(10), 'sigma':0.15*np.log(10)})
    prior.append({'name':'p',     'index':0, 'type':'n',  'mu':1.05,            'sigma':0.13})
    prior.append({'name':'c',     'index':0, 'type':'ln', 'mu':-4.02,           'sigma':1.42})
    prior.append({'name':'sigma', 'index':0, 'type':'ln', 'mu':np.log(0.2),     'sigma':1.0})
    return prior

def prior_generic_gr():
    prior =[]
    prior.append({'name':'beta',  'index':0, 'type':'n',  'mu':0.85*np.log(10), 'sigma':0.15*np.log(10)})
    prior.append({'name':'sigma', 'index':0, 'type':'ln', 'mu':np.log(0.2),     'sigma':1.0})
    return prior

def para_generic():
    para = {'beta':0.85*np.log(10),'k':np.exp(-4.86),'p':1.05,'c':np.exp(-4.02)}
    return para

#######################################################################
## forecast model 
#######################################################################
class OU_GRDR():
    
################
    def mcmc(self,Data,itv,n_sample=12500,n_core=8,prior=prior_generic(),opt=[],dilute=100):
        mag_ref = Data["Mag"][0]
        self.fit(Data,itv,mag_ref,prior=prior,opt=['ste'])
        logging.info('  start MCMC sampling')
        [para_mcmc,L_mcmc,dtl_mcmc] = MCMC(self,n_sample,n_core,prior=prior,opt=opt)
        para_mcmc = para_mcmc[::dilute]
        L_mcmc = L_mcmc[::dilute]
        self.para_mcmc = para_mcmc
        self.L_mcmc = L_mcmc
        self.para_mle = para_mcmc.iloc[0]
        self.L_mle = L_mcmc[0]
        return {'para_mcmc':para_mcmc,'L_mcmc':L_mcmc,'para_mle':para_mcmc.iloc[0],'L_mle':L_mcmc[0],'itv':itv,'mag_ref':mag_ref,'mu':self.mu,'mu0':self.mu0,'ste':self.ste,'prior':prior,'dtl_mcmc':dtl_mcmc}

    def fit(self,Data,itv,mag_ref,prior=prior_generic(),opt=[]):
        logging.info('  fitting a model')
        st,en = itv
        T = Data['T']; Mag = Data['Mag'];
        index = ( st + 1e-6 < T ) & ( T < en - 1e-6 )
        self.T = T[index].values.copy()
        self.Mag = Mag[index].values.copy()
        n = self.T.shape[0]
        self.n = n
        self.mag_ref = mag_ref
        self.itv = itv
        self.itv_iei = [np.append(st,self.T),np.append(self.T,en)]
        param_gr = GRDR_SSM().fit(Data,itv,prior=prior_generic_gr(),opt=[])
        self.mu0 = param_gr['mu']
        self.mu0_ext = np.append(self.mu0,self.mu0[-1])

        stg = {}
        stg['para_list']      = ['beta','mu1','sigma','k','p','c']
        stg['para_length']    = {'beta':1,'mu1':1,'sigma':1,'k':1,'p':1,'c':1}
        stg['para_exp']       = {'beta':True, 'mu1':False,'sigma':True,'k':True,  'p':True, 'c':True}
        stg['para_ini']       = {'beta':1.8,  'mu1':0.0,  'sigma':0.2,  'k':100.0, 'p':1.11, 'c': 1e-2}
        stg['para_step_Q']    = {'beta':0.2,  'mu1':0.1,  'sigma':0.2,  'k':1.0,   'p':0.1,  'c': 0.3}
        stg['para_step_diff'] = {'beta':0.01, 'mu1':0.01, 'sigma':0.05, 'k':0.05,  'p':0.01, 'c': 0.05}
        self.stg = stg
        
        self.para = stg['para_ini']
        self.only_L = True
        stg['para_ini']['beta'] = param_gr['para']['beta']
        stg['para_ini']['sigma'] = param_gr['para']['sigma']
        stg['para_ini']['k'] *= n / self.Int_OU_GRDR()[0].sum()
        
        logging.info('  -- estimate ou_grdr')
        [para,L,ste,_,_] = Quasi_Newton(self,prior=prior,opt=opt)
        self.para = para
        self.L = L
        self.ste = ste
        self.mu = self.mu0 + self.para['mu1']
        
        return {'para':para, 'mu':self.mu, 'mu0':self.mu0, 'L':L, 'ste':ste, 'itv':itv, 'mag_ref':mag_ref, 'prior':prior}
            
    def LG(self,para,only_L=False):
        self.para = para
        self.only_L = only_L
        
        [Sum,d_Sum] = self.Sum_OU_GRDR()
        [Int,d_Int] = self.Int_OU_GRDR()
        
        L = Sum - Int
        G = { key:d_Sum[key] - d_Int[key] for key in d_Sum }
        
        return [L,G]

    def Sum_OU_GRDR(self):
        [sum_OU,   d_sum_OU  ] = self.Sum_OU()
        [sum_GRDR, d_sum_GRDR] = self.Sum_GRDR()
        
        Sum = sum_OU + sum_GRDR

        d_Sum = {}
        if not self.only_L:
            d_Sum['beta']  = d_sum_GRDR['beta']
            d_Sum['mu1']   = d_sum_GRDR['mu1']
            d_Sum['sigma'] = d_sum_GRDR['sigma']
            d_Sum['k']     = d_sum_OU['k']
            d_Sum['p']     = d_sum_OU['p']
            d_Sum['c']     = d_sum_OU['c']

        return [Sum,d_Sum]

    def Int_OU_GRDR(self):
        [int_GRDR, d_int_GRDR] = self.Int_GRDR()
        [int_OU,   d_int_OU  ] = self.Int_OU()

        Int = ( int_OU * int_GRDR ).sum()

        d_Int = {}
        if not self.only_L:
            d_Int['beta']  = ( int_OU        * d_int_GRDR['beta']  ).sum()
            d_Int['mu1']   = ( int_OU        * d_int_GRDR['mu1']   ).sum()
            d_Int['sigma'] = ( int_OU        * d_int_GRDR['sigma'] ).sum()
            d_Int['k']     = ( d_int_OU['k'] * int_GRDR            ).sum()
            d_Int['p']     = ( d_int_OU['p'] * int_GRDR            ).sum()
            d_Int['c']     = ( d_int_OU['c'] * int_GRDR            ).sum()

        return [Int,d_Int]
    
    def Sum_GRDR(self):
        beta = self.para['beta']; mu = self.mu0 + self.para['mu1']; sigma = self.para['sigma'];
        mag_ref = self.mag_ref
        Mag = self.Mag
        n_Mag = (Mag-mu)/sigma

        Sum = ( np.log(beta) - beta*(Mag-mag_ref) + np.log( norm.cdf(n_Mag) ) ).sum()

        d_Sum = {}
        if not self.only_L:
            d_Sum['beta']  = ( 1.0/beta - (Mag-mag_ref)                                        ).sum()
            d_Sum['mu1']   = (                          +  (-1.0/  sigma)*d_Lnormcdf(n_Mag) ).sum()
            d_Sum['sigma'] = (                          +  (-n_Mag/sigma)*d_Lnormcdf(n_Mag) ).sum()

        return [Sum,d_Sum]
    
    def Int_GRDR(self):
        beta = self.para['beta']; mu = self.mu0_ext + self.para['mu1']; sigma = self.para['sigma'];
        mag_ref = self.mag_ref
        
        Int = np.exp( beta*(mag_ref-mu) + (beta*sigma)**2.0/2.0 )

        d_Int = {}
        if not self.only_L:
            d_Int['beta']  = Int * ( (mag_ref-mu) + beta*sigma**2.0 )
            d_Int['mu1']   = Int * ( -beta                          )
            d_Int['sigma'] = Int * (                beta**2.0*sigma )

        return [Int,d_Int]
    
    def Sum_OU(self):
        k = self.para['k']; p = self.para['p']; c = self.para['c'];
        T = self.T
        n = self.n
        
        Sum = ( np.log(k) - p*np.log(T+c) ).sum()

        d_Sum = {}
        if not self.only_L:
            d_Sum['k'] = n/k
            d_Sum['p'] = ( -np.log(T+c) ).sum()
            d_Sum['c'] = ( -p/(T+c)     ).sum()


        return [Sum,d_Sum]
    
    def Int_OU(self):        
        k = self.para['k']; p = self.para['p']; c = self.para['c'];
        st,en = self.itv_iei;
        
        f_st = (st+c)**(-p+1.0)/(-p+1.0)
        f_en = (en+c)**(-p+1.0)/(-p+1.0)
        
        Int = k * ( f_en - f_st )
        
        d_Int = {}
        if not self.only_L:
            d_Int['k'] = (f_en-f_st)
            d_Int['p'] = -k * ( np.log(en+c)*f_en - np.log(st+c)*f_st ) + k * (f_en-f_st) /(-p+1.0)
            d_Int['c'] = k * ( (en+c)**(-p) - (st+c)**(-p) )

        return [Int,d_Int]
    
################
    def predict(self,itv_fore):
        logging.info('  predict [%.3f,%.3f]' % (itv_fore[0],itv_fore[1]) )
        para_mcmc = self.para_mcmc
        mag_ref = self.mag_ref

        [mag,exp_num] = self.expected_cumulative_number(para_mcmc,mag_ref,itv_fore)
        exp_num_generic = self.expected_cumulative_number(para_generic(),mag_ref,itv_fore)[1].flatten()

        pred = [ self.bayesian_forecasting(l) for l in exp_num ]
        pred = pd.DataFrame(pred)
        pred['mag'] = mag
        #pred['exp_num_generic'] = exp_num_generic

        return {'pred':pred, 'itv':itv_fore, 'para':para_mcmc.iloc[0], 'mag_ref':mag_ref}
    
    @staticmethod
    def predict_from_param(itv_fore,param):
        model = OU_GRDR()
        model.para_mcmc = param['para_mcmc']
        model.mag_ref   = param['mag_ref']
        return model.predict(itv_fore)
    
    @staticmethod
    def expected_cumulative_number(para,mag_ref,itv_fore):
        k    = np.array(para['k']).reshape(1,-1)
        p    = np.array(para['p']).reshape(1,-1)
        c    = np.array(para['c']).reshape(1,-1)
        beta = np.array(para['beta']).reshape(1,-1)    
        st,en = itv_fore

        mag = np.arange(0.95,10.0,0.1).reshape(-1,1);

        c_OU = k * ( (en+c)**(-p+1.0) - (st+c)**(-p+1.0) ) / (-p+1.0)
        c_GR = np.exp(  beta*(mag_ref-mag) )
        c_cum_exp = c_GR * c_OU

        return [mag.flatten(),c_cum_exp]
    
    @staticmethod
    def bayesian_forecasting(l):
        x_min = -1
        x_max = 72 if l.max()<36.0 else ( np.ceil(  l.max() + np.sqrt(l.max())*6.0 ) ).astype('i')

        while poisson.cdf(x_max,l).mean() < 0.975:
            x_max *= 2

        def bisection_method(x0,x2,p):

            while 1:
                x1 = np.rint((x0+x2)/2).astype('i8')
                y1 = poisson.cdf(x1,l).mean()

                if y1 < p:
                    x0 = x1
                else:
                    x2 = x1

                if (x2-x0) == 1:
                    break

            return x2

        error_l = bisection_method(x_min,x_max,0.025)
        error_u = bisection_method(x_min,x_max,0.975)

        p_nzs = 1.0 - np.exp(-l)
        p_nz = np.mean(p_nzs)
        p_nz_map = p_nzs[0]
        p_nz_25 = np.percentile(p_nzs,25,interpolation="midpoint")
        p_nz_50 = np.percentile(p_nzs,50,interpolation="midpoint")
        p_nz_75 = np.percentile(p_nzs,75,interpolation="midpoint")

        #pred_s = {'exp_num':l[0], 'error_l':error_l, 'error_u':error_u, 'p_nz':p_nz, 'p_nz_map':p_nz_map, 'p_nz_25':p_nz_25, 'p_nz_50': p_nz_50, 'p_nz_75':p_nz_75}
        
        pred_s = {
            "expected_number": l[0],
            "lower_bound_95%_interval": error_l,
            "upper_bound_95%_interval": error_u,
            "probability_of_one_or_more_events": p_nz,
        }

        return pred_s
    
    @staticmethod
    def add_obs_num(fore,Data,catalog_test='observed'):
        logging.info('  add obs_num')
        pred = fore['pred']; st,en = fore['itv'];
        T = Data['T'].values
        Mag = Data['Mag'].values
        Mag = Mag[( st+1e-6 < T ) & ( T < en-1e-6 )]
        n_hist = np.histogram(Mag,np.hstack([pred['mag'].values,np.inf]))[0]
        obs_num = n_hist[::-1].cumsum()[::-1]
        pred['observed_number'] = obs_num
        return fore
    
#######################################################################
class GRDR_SSM:
    
    def fit(self,Data,itv,prior=[],opt=[]):
        logging.info('  -- estimate detection rate')
        T = Data['T']
        Mag = Data['Mag']
        [st,en] = itv
        index = ( st + 1e-6 < T ) & ( T < en - 1e-6 )
        Mag = Mag[index].values.copy()
        
        mu = Mag.mean() * np.ones_like(Mag)
        n = mu.shape[0]
        [W,rank_W] = self.Weight_Matrix_2nd(n)
        
        self.Mag = Mag
        self.mu = mu
        self.W = W
        self.rank_W = rank_W
        
        stg = {}
        stg['para_list']      = ['beta','sigma','V'] 
        stg['para_length']    = {'beta':1,    'sigma':1,     'V':1}
        stg['para_exp']       = {'beta':True, 'sigma':True, 'V':True }
        stg['para_ini']       = {'beta':2.0,  'sigma':0.2,   'V':1e-6  }
        stg['para_step_Q']    = {'beta':0.2,  'sigma':0.2,   'V':0.2  }
        stg['para_step_diff'] = {'beta':0.01, 'sigma':0.01,  'V':0.01 }
        self.stg = stg
        
        [para,L,ste,_,_] = Quasi_Newton(self,prior=prior,opt=opt)
        self.para = para
        self.Estimate_mu()
        mu = self.mu
        
        return {'para':para,  'mu':mu, 'L':L, 'ste':ste, 'itv':itv, 'prior':prior}
    
    def LG(self,para,only_L=False):   
        beta = para['beta']; sigma = para['sigma']; V = para['V'];

        #Likelihood
        L = self.M_L(para)

        #Gradient
        G = {}
        epsilon = {'beta':beta*0.01,'sigma':sigma*0.01,'V':V*0.01}

        for para_name in para:

            para1 = {'beta':beta, 'sigma':sigma, 'V':V};  para1[para_name] = para1[para_name] - 2.0*epsilon[para_name];  L1 = self.M_L(para1);
            para2 = {'beta':beta, 'sigma':sigma, 'V':V};  para2[para_name] = para2[para_name] - 1.0*epsilon[para_name];  L2 = self.M_L(para2);
            para3 = {'beta':beta, 'sigma':sigma, 'V':V};  para3[para_name] = para3[para_name] + 1.0*epsilon[para_name];  L3 = self.M_L(para3);
            para4 = {'beta':beta, 'sigma':sigma, 'V':V};  para4[para_name] = para4[para_name] + 2.0*epsilon[para_name];  L4 = self.M_L(para4);

            G[para_name] = ( L1 - 8.0*L2 + 8.0*L3 - L4 )/12.0/epsilon[para_name]

            """
            para1 = {'beta':beta, 'sigma':sigma, 'V':V}; para1[para_name] = para1[para_name] - 1.0*epsilon[para_name]; L1  = M_L(para1);
            para2 = {'beta':beta, 'sigma':sigma, 'V':V}; para2[para_name] = para2[para_name] + 1.0*epsilon[para_name]; L2  = M_L(para2);

            G[para_name] = ( L2 - L1 )/2.0/epsilon[para_name]
            """

        return [L,G]
    
    def M_L(self,para):
        self.para = para
        V = para['V']
        n = self.Mag.shape[0]
        [L,G,H] = self.Estimate_mu()
        LU = spla.splu(-H)
        log_det = np.log(np.abs(LU.U.diagonal())).sum()
        ML = L + np.log(2.0*np.pi)*n/2.0 - log_det/2.0
        ML = ML - np.exp(-(V-5e-8)/1e-8)
        return ML
        
    def Estimate_mu(self):

        while 1:
            [L,G,H]=self.LGH()

            if np.linalg.norm(G)<1e-5:
                break

            d = -spla.spsolve(H,G)

            d_max = np.abs(d).max()
            if d_max > 1.0:
                d = d/d_max

            self.mu = self.mu + d

        return [L,G,H]
    
    def LGH(self):
        Mag = self.Mag
        beta = self.para['beta']; sigma = self.para['sigma']; V = self.para['V'];
        mu = self.mu
        W = self.W; rank_W = self.rank_W;
        n = Mag.shape[0]

        ##LGH
        n_Mag=(Mag-mu)/sigma
        L = ( np.log(beta) -beta*(Mag-mu) -(beta*sigma)**2.0/2.0 +np.log(norm.cdf(n_Mag)) ).sum() - rank_W*np.log(2*np.pi*V)/2.0 - mu.dot(W.dot(mu))/2.0/V
        G = ( beta + d_Lnormcdf(n_Mag)*(-1.0/sigma) ) - W.dot(mu)/V
        H = sparse.csc_matrix(sparse.spdiags(d2_Lnormcdf(n_Mag)*(1.0/sigma**2.0),0,n,n)) - W/V

        return [L,G,H]
    
    @staticmethod
    def Weight_Matrix_2nd(n):

        d0 = np.hstack(([1,5],6*np.ones(n-4),[5,1]))
        d1 = np.hstack((-2,-4*np.ones(n-3),-2))
        d2 = np.ones(n-2)

        data = [d2,d1,d0,d1,d2]
        diags = np.arange(-2,3)
        W = sparse.diags(data,diags,shape=(n,n),format='csc')

        rank_W = n-2

        return [W,rank_W]

def d_Lnormcdf(x):
    return norm.pdf(x)/norm.cdf(x)

def d2_Lnormcdf(x):
    a=d_Lnormcdf(x)
    return -a*(a+x)

