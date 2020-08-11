
"""
Created on Wed Sep 11 09:42:57 2019

@author: victor.long
"""

#%% import libs

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import t as tdist
from scipy.stats import norm 
from scipy.stats import kendalltau

from matplotlib import pyplot as plt

from rpy2.robjects import r, numpy2ri, pandas2ri
numpy2ri.activate()
pandas2ri.activate()


#%% RiskModels Class

class RiskModels(object):
    """
    A tool for estimating univariate variance or multivariate
    covariance/correlation matrix;
    """
    def __init__(self, X, XType='return'):
        # check inputs
        assert (isinstance(X, pd.DataFrame)), \
            'Please use DataFrame for X!'

        assert (XType in ['return', 'price']), \
            "XType should be 'return' or 'price'!"

        if XType == 'price':
            X = (X/X.shift(1)-1).drop(X.index[0])

        # initial inputs
        self._X = X

        # result fields
        self._residuals = None
        self._stds = None
        self._corrs = None
        self._paras = {}
        self._model = {}
        self._val = None
        self._Q = None  # t+1 forecast of Q to facilitate correlation forecasts 

#%% Show Univariate ACF or GARCH Results

    def _plot_acf(self, acf_x, pacf_x, acf_x2, pacf_x2, name, y, model):
        """
        Make ACF and PACF plots as R;
        95% CI is calculated as ±1.96/sqrt(n);
        """
        n = self._X.shape[0]
        ci = 1.96/np.sqrt(n)

        fig = plt.figure()

        # ACF of x
        acf = acf_x.rx2('acf').flatten()
        lag = acf_x.rx2('lag').flatten()
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axhline(0, linewidth=0.5, color='xkcd:black')
        ax1.axhline(ci, linewidth=1, linestyle='--', color='xkcd:salmon')
        ax1.axhline(-ci, linewidth=1, linestyle='--', color='xkcd:salmon')
        plt.bar(lag, acf, width=0.1, color='xkcd:cerulean')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('ACF')
        ax1.set_title(f"{name} | {y} | {model} model", fontsize=10)

        # PACF of x
        acf = pacf_x.rx2('acf').flatten()
        lag = pacf_x.rx2('lag').flatten()
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axhline(0, linewidth=0.5, color='xkcd:black')
        ax2.axhline(ci, linewidth=1, linestyle='--', color='xkcd:salmon')
        ax2.axhline(-ci, linewidth=1, linestyle='--', color='xkcd:salmon')
        plt.bar(lag, acf, width=0.1, color='xkcd:cerulean')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('PACF')
        ax2.set_title(f"{name} | {y} | {model} model", fontsize=10)

        # ACF of x**2
        acf = acf_x2.rx2('acf').flatten()
        lag = acf_x2.rx2('lag').flatten()
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.axhline(0, linewidth=0.5, color='xkcd:black')
        ax3.axhline(ci, linewidth=1, linestyle='--', color='xkcd:salmon')
        ax3.axhline(-ci, linewidth=1, linestyle='--', color='xkcd:salmon')
        plt.bar(lag, acf, width=0.1, color='xkcd:blue green')
        ax3.set_xlabel('Lag')
        ax3.set_ylabel('ACF')
        ax3.set_title(f"{name} | {y}^2 | {model} model", fontsize=10)

        # PACF of x**2
        acf = pacf_x2.rx2('acf').flatten()
        lag = pacf_x2.rx2('lag').flatten()
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axhline(0, linewidth=0.5, color='xkcd:black')
        ax4.axhline(ci, linewidth=1, linestyle='--', color='xkcd:salmon')
        ax4.axhline(-ci, linewidth=1, linestyle='--', color='xkcd:salmon')
        plt.bar(lag, acf, width=0.1, color='xkcd:blue green')
        ax4.set_xlabel('Lag')
        ax4.set_ylabel('PACF')
        ax4.set_title(f"{name} | {y}^2 | {model} model", fontsize=10)

        fig.tight_layout()
        plt.show()

    def showACF(self, name=None, col=0, model=None):
        """
        Show univariate ACF and PACF plot;
        Use col to specify data if name is not given;
        Model==False means using original data;
        """
        if name:
            assert(name in self._X.columns), "Name given is not recognized!"
        else:
            name = self._X.columns[col]
        
        assert (model in self._paras.keys() or model is None), \
            "Unrecognized model name or model hasn't been fitted yet!"
        
        # whether check residuals or original data
        if model:
            x = self.residuals(model=model)[name]
            y = 'Residuals'
        else:
            x = self._X[name]
            y = 'Returns'

        # import R functions
        # r.par(mfrow=r.c(2, 2))
        # r.acf(x, main=f"ACF | {name} | {y} | {model} model")
        # r.pacf(x, main=f"PACF | {name} | {y} | {model} model")
        # r.acf(x**2, main=f"ACF | {name} | {y}^2 | {model} model")
        # r.pacf(x**2, main=f"PACF | {name} | {y}^2 | {model} model")

        # a workaround to use own plot function for mac
        a = r.acf(x, pl=False)
        b = r.pacf(x, pl=False)
        c = r.acf(x**2, pl=False)
        d = r.pacf(x**2, pl=False)
        self._plot_acf(a, b, c, d, name, y, model)

    @staticmethod
    def _checkgp(model, garch, arma, archpow, dist):
        """
        Private function to check GARCH parameters
        """
        assert (model in ['s', 'gjr', 'm', 'e']), \
            "Model should be 's', 'gjr', 'm' or 'e'!"

        assert (len(garch) == 2), \
            "GARCH should have only 2 parameters!"

        assert (len(arma) == 2), \
            "ARMA should have only 2 parameters!"

        am = False
        if model == 'm':
            assert (archpow in [1, 2]), \
                "GARCH-M model should have parameter archpow of 1 or 2!"
            am = True
            model = 's'

        assert (dist in ['norm', 'snorm', 'std', 'sstd']), \
            "distribution should be 'norm','snorm','std' or 'sstd'!"

        v = model + 'GARCH'
        gOrder = r.c(garch[0], garch[1])
        aOrder = r.c(arma[0], arma[1])
        apow = archpow
        return v, gOrder, aOrder, am, apow, dist

    def showGARCH(self, name=None, col=0, model='s', garch=(1, 1), arma=(1, 1),
                  archpow=1, dist='norm'):
        """
        Show univariate GARCH with ARMA;
        models: standard garch; e-garch; gjr-garch; garch-m;
        garch are parameters for GARCH;
        arima should be parameters for arima or None;
        dist: normal; skewed normal; student-t; skewed student-t;
        """

        # model parameters
        v, gOrder, aOrder, am, apow, dist = \
            self._checkgp(model, garch, arma, archpow, dist)

        # data
        if name:
            assert(name in self._X.columns), "Name given is not recognized!"
        else:
            name = self._X.columns[col]
        x = self._X[name]

        # import rugarch
        r.require("rugarch")

        # model specifics
        spec = r.ugarchspec(

                variance_model=r.list(model=v,
                                      garchOrder=gOrder
                                      ),

                mean_model=r.list(armaOrder=aOrder,
                                  archm=am,
                                  archpow=apow
                                  ),

                distribution_model=dist
                )

        # fit and show
        fit = r.ugarchfit(data=x, spec=spec)
        print(fit)

#%% GARCH Model

    def fitGARCH(self, model='s', garch=(1, 1), arma=(1, 1),
                 archpow=1, dist='norm'):
        """
        fit all columns to GARCH model;
        generate standardized residuals;
        """

        # model parameters
        v, gOrder, aOrder, am, apow, dist = \
            self._checkgp(model, garch, arma, archpow, dist)

        # import rugarch
        r.require("rugarch")

        # model specifics
        spec = r.ugarchspec(

                variance_model=r.list(model=v,
                                      garchOrder=gOrder
                                      ),

                mean_model=r.list(armaOrder=aOrder,
                                  archm=am,
                                  archpow=apow
                                  ),

                distribution_model=dist
                )

        n = self._X.shape[1]
        residuals = np.empty(self._X.shape)
        stds = np.empty(self._X.shape)
        
        # number of mean model's parameters
        m = sum(arma) + 1 + am  
        
        # number of variance model's parameters
        vn = sum(garch) + 1 + (dist != 'norm') + (dist == 'sstd')
        
        para_mean = np.empty([m, n])  # para of mean model
        para_var = np.empty([vn, n])  # para of var model

        # fit data
        fit = None  # define fit to eliminate Pycharm warning
        for i in range(n):

            # try:

            name = self._X.columns[i]
            x = self._X.copy()[name]
            fit = r.ugarchfit(data=x, spec=spec)

            # store standardized garch residuals
            residuals[:, i] = r.residuals(fit, standardize=True).flatten()

            # store standard deviations and parameters
            stds[:, i] = r.sigma(fit).flatten()
            para_mean[:, i] = r.coef(fit).flatten()[:m]
            para_var[:, i] = r.coef(fit).flatten()[m:]

            # except ValueError:
            #     print(f"The column {i} is not converged!")
            #     return

        # update results
        self._residuals = residuals
        self._stds = stds
        if am:  # check if garch-in-mean model
            extra = '-m'+str(apow)
        else:
            extra = ''
        self._model['mean'] = 'ARMA'+str(arma)+extra
        self._model['variance'] = v+str(garch)
        self._model['variance distribution'] = dist

        # a workaround for mac in case names missing
        numpy2ri.deactivate()
        pandas2ri.deactivate()
        idx = np.array(r.names(r.coef(fit)))
        numpy2ri.activate()
        pandas2ri.activate()

        cols = self._X.columns
        self._paras['mean'] = \
            pd.DataFrame(para_mean, index=idx[:m], columns=cols)
        self._paras['variance'] = \
            pd.DataFrame(para_var, index=idx[m:], columns=cols)

#%% GARCH-DCC Model

    def _Q_bar(self):
        """
        Private function to calculate Q_bar in DCC
        """
        T = self._residuals.shape[0]
        Q_bar = 0
        for i in range(T):
            eps = self._residuals[i, :].reshape(-1, 1)
            Q_bar += eps.dot(eps.T)
        return Q_bar/T

    @staticmethod
    def _Qt(a, b, Q_bar, epst_1, Qt_1):
        """
        Private function to calculate Qt matrix in DCC;
        """
        return (1-a-b)*Q_bar + a*epst_1.dot(epst_1.T) + b*Qt_1

    @staticmethod
    def _Rt(Qt):
        """
        Private function to calculate Dynamic correlation
        """
        Qt_star = np.sqrt(np.diag(np.diag(Qt)))
        Qt_star_inv = np.linalg.inv(Qt_star)
        return Qt_star_inv.dot(Qt).dot(Qt_star_inv)

    def _mle_norm(self, para):
        """
        Objective function to max MLE of gaussian DCC
        """
        T = self._residuals.shape[0]
        n = self._residuals.shape[1]
        eps = np.zeros([n, 1])  # initialize eps_0 as zero
        Q_bar = self._Q_bar()
        # Q = np.eye(n)  # initialize Q_0 as diagonal ones
        Q = np.copy(Q_bar)  # initialize Q_0 as Q_bar

        a = para[0]
        b = para[1]

        mle = 0
        for i in range(T):
            Q = self._Qt(a, b, Q_bar, eps, Q)
            R = self._Rt(Q)
            eps = self._residuals[i, :].reshape(-1, 1)
            mle += np.log(np.linalg.det(R)) + eps.T.dot(np.linalg.inv(R)).dot(eps)
        return mle

    def _mle_t(self, para):
        """
        Objective function to max MLE of student-t DCC
        """
        T = self._residuals.shape[0]
        n = self._residuals.shape[1]
        eps = np.zeros([n, 1])  # initialize eps_0 as zero
        Q_bar = self._Q_bar()
        Q = np.copy(Q_bar)  # initialize Q_0 as Q_bar

        a = para[0]
        b = para[1]
        v = para[2]

        mle = 0
        for i in range(T):
            Q = self._Qt(a, b, Q_bar, eps, Q)
            R = self._Rt(Q)
            eps = self._residuals[i, :].reshape(-1, 1)
            mle += (
                    np.log(gamma((v+n)/2)) -
                    np.log(gamma(v/2)) -
                    n/2*np.log(np.pi*(v-2)) -
                    1/2*np.log(np.linalg.det(R)) -
                    (v+n)/2*np.log(1+eps.T.dot(np.linalg.inv(R)).dot(eps)/(v-2))
            )

        return -mle
    
    @staticmethod
    def _omega(v, sigma):
        """
        Private function to calculate dispersion matrix in multivariate skew t
        """
        K = np.sqrt(
                    1 +
                    4*v*(v-2) * 
                    (np.pi*gamma(v/2)**2-(v-2)*gamma((v-1)/2)**2) * 
                    sigma.T.dot(sigma) / 
                    (np.pi*gamma(v/2)**2*(v-(v-2)*sigma.T.dot(sigma))**2)
        )
        
        n = sigma.shape[0]
        In = np.eye(n)  # identity matrix
        Zeros = np.zeros([n, 1])
        
        if np.all(sigma == Zeros):
            omega = (v-2)/v*In
        else:
            omega = (v-2)/v * (
                    In + 1/(sigma.T.dot(sigma)) * (
                        -1 + np.pi*gamma(v/2)**2*(v-(v-2)*sigma.T.dot(sigma)) / (
                            2*sigma.T.dot(sigma)*(v-2) *
                            (np.pi*gamma(v/2)**2-(v-2)*gamma((v-1)/2)**2)
                        ) * (-1+K)
                    ) * sigma.dot(sigma.T)
            )
        return omega
    
    @staticmethod
    def _xi(v, sigma, omega):
        """
        Private function to calculate location vector in multivariate skew t
        """
        xi = -1 * np.sqrt(v/np.pi) * gamma((v-1)/2)/gamma(v/2) * \
            omega.dot(sigma) / np.sqrt(1+sigma.T.dot(omega).dot(sigma))
        return xi
        
    @staticmethod 
    def _D(omega):
        """
        Private function to calculate D in multivariate skew t
        """
        return np.sqrt(np.diag(np.diag(omega)))
        
    @staticmethod
    def _delta(sigma, D):
        """
        Private function to calculate skewness vector in multivariate skew t
        """
        return D.dot(sigma)
    
    @staticmethod
    def _Dt(stds):
        """
        Private function to calculate diagonal stds in multivariate skew t
        """
        return np.diag(stds)
    
    @staticmethod    
    def _Ht_sqrt_inv(Dt, Rt):
        """
        Private function to calculate the inverse of 
        Cholesky decomposition of cov matrix in multivariate skew t
        """
        Ht_sqrt = np.linalg.cholesky(Dt.dot(Rt).dot(Dt))  
        return np.linalg.inv(Ht_sqrt)      
    
    @staticmethod  
    def _Qat(H, at, xi, omega):
        """
        Private function to calculate Qat matrix in multivariate skew t
        """
        x = H.dot(at)-xi
        return x.T.dot(np.linalg.inv(omega)).dot(x)
    
    def _mle_skewt(self, para):
        """
        Objective function to max MLE of skew-t DCC
        """
        T = self._residuals.shape[0]
        n = self._residuals.shape[1]
        eps = np.zeros([n, 1])  # initialize eps_0 as zero
        Q_bar = self._Q_bar()
        Q = np.copy(Q_bar)  # initialize Q_0 as Q_bar
        
        a = para[0]
        b = para[1]
        v = para[2]
        s = para[3::].reshape(-1, 1)  # sigma vector
        
        omega = self._omega(v, s)
        D = self._D(omega)
        delta = self._delta(s, D)
        xi = self._xi(v, s, omega)
     
        mle = 0
        for i in range(T):
            Q = self._Qt(a, b, Q_bar, eps, Q)
            R = self._Rt(Q)
            eps = self._residuals[i, :].reshape(-1, 1)
            
            std = self._stds[i, :]
            Dt = self._Dt(std)
            H = self._Ht_sqrt_inv(Dt, R)
            at = eps*std.reshape(-1, 1)
            Qa = self._Qat(H, at, xi, omega)
            
            t = delta.T.dot(np.linalg.inv(D)).dot(H.dot(at)-xi)/np.sqrt(
                (v+n)/(Qa+v))  # input value of student-t dist cdf
            
            mle += (
                    np.log(gamma((v+n)/2)) -
                    1/2*np.log(np.linalg.det(omega)) - 
                    n/2*np.log(np.pi*v) -
                    np.log(gamma(v/2)) -
                    (v+n)/2*np.log(1+Qa/v) -
                    np.log(tdist.cdf(t, v+n)) -
                    1/2*np.log(np.linalg.det(R)) 
            )

        return -mle
        
    def fitDCC(self, model='s', garch=(1, 1), arma=(1, 1),
               archpow=1, dist='norm', dist_dcc='norm'):
        """
        Fit GARCH-DCC(1,1) model;
        Multivariate normal and student-t available;
        See details in 'Multivariate DCC-GARCH Model - With Various 
        Error distributions';
        """
        assert (dist_dcc in ['norm', 'std', 'sstd']), \
            "dist_dcc should be 'norm', 'std' or 'sstd'!"

        # fit garch
        self.fitGARCH(model, garch, arma, archpow, dist)

        # constraints: a,b>0; a+b<1
        bnds = [(0, None), (0, None)]
        const = {'type': 'ineq', 'fun': lambda para: 1-para[0]-para[1]}

        # objective function of DCC mle and starting values
        if dist_dcc == 'norm':
            obj = self._mle_norm
            x_0 = np.array([0.2, 0.2])
            idx = ['a', 'b']  # parameter names
            
        elif dist_dcc == 'std':
            obj = self._mle_t
            x_0 = np.array([0.2, 0.2, 10])
            bnds.append((2, None))  # v should be larger than 2
            idx = ['a', 'b', 'nu']
            
        else:
            obj = self._mle_skewt
            n = self._X.shape[1]
            x_0 = np.array([0.2, 0.2, 10] +
                           [0.1]*n)  # sigma initialize as 0.1
            bnds.append((2, None))  # v should be larger than 2
            bnds.extend([(None, None)]*n)  # no boundaries for sigma vector
            idx = ['a', 'b', 'nu'] + ['sig'+str(i) for i in range(1, n+1)]

        # fit DCC
        result = minimize(fun=obj,
                          x0=x_0,
                          method='SLSQP',
                          constraints=const,
                          bounds=bnds)

        # store results
        self._val = result.fun
        self._model['correlation'] = 'DCC(1, 1)'
        self._model['correlation distribution'] = dist_dcc
        self._paras['correlation'] = pd.Series(result.x, index=idx)

        # warning if not converged
        if not result.success:
            print('\n')
            print('-'*50)
            print('DCC is not converged! \n')
            print('-'*50)

        # calculate correlations
        n = self._X.shape[1]
        T = self._X.shape[0]
        a = result.x[0]  # parameters
        b = result.x[1]

        eps = np.zeros([n, 1])  # initialize eps_0 as zero
        Q_bar = self._Q_bar()
        Q = np.copy(Q_bar)  # initialize Q_0 as Q_bar

        corrs = np.empty([n, n, T])  # container of correlations
        for i in range(T):
            Q = self._Qt(a, b, Q_bar, eps, Q)
            R = self._Rt(Q)
            corrs[:, :, i] = R
            eps = self._residuals[i, :].reshape(-1, 1)

        self._corrs = corrs
        self._Q = self._Qt(a, b, Q_bar, eps, Q)  # t+1 of Q 
        
#%% GARCH-DCC-Copula model
    
    @staticmethod
    def _marginals(data, marginal):
        """
        Estimation of marginal cumulative density of each variable;
        """    
        d = data.shape[1]  # number of assets
        n = data.shape[0]  # number of periods
        cd = np.empty([n, d])
        
        if marginal == 'empirical':
            for i in range(d):
                for j in range(n):
                    cd[j, i] = np.sum(data[:, i] <= data[j, i]) / (n+1)
            return cd, None    
        
        elif marginal == 'norm':
            paras = np.empty([2, d])  # store parameters
            for i in range(d):
                mu, std = norm.fit(data[:, i])
                paras[:, i] = [mu, std]
                for j in range(n):
                    cd[j, i] = norm.cdf(data[j, i], loc=mu, scale=std)
            return cd, paras
        
        elif marginal == 'std':
            paras = np.empty([3, d])  # store parameters
            for i in range(d):
                df, mu, std = tdist.fit(data[:, i])
                paras[:, i] = [df, mu, std]
                for j in range(n):
                    cd[j, i] = tdist.cdf(data[j, i], df=df, loc=mu, scale=std)
            return cd, paras
    
    @staticmethod
    def _P(U):
        """
        An empirical estimate of correlation matrix P by estimating empirical 
        Kendall’s tau for each bivariate margin of the t copula;
        """
        n = U.shape[1]
        tau = np.empty([n, n])  # container of Kendall's tau
        
        # upper triangle of tau
        for i in range(n-1):
            for j in range(i+1, n):
                tau[i, j] = kendalltau(U[:, i], U[:, j])[0]
                
        # fill lower triangle of tau
        for i in range(1, n):
            for j in range(i):
                tau[i, j] = tau[j, i]
                
        # fill diagonal of tau        
        for i in range(n):
            tau[i, i] = 1
        
        return np.sin(np.pi/2*tau)
    
    @staticmethod    
    def _t_pdf(nu, P, X):
        """
        Probability density of a standardized multivariate t distribution 
        with d.o.f of nu, mean of zero, and correlation matrix of P;
        """
        p = X.shape[0]  # dimensions
        pdf = gamma((nu+p)/2) / (
               gamma(nu/2)*nu**(p/2)*np.pi**(p/2)*np.sqrt(np.linalg.det(P))
               ) * (1+1/nu*X.T.dot(np.linalg.inv(P)).dot(X))**(-(nu+p)/2)
        return pdf
    
    def _mle_t_copula(self, nu, *args):
        """
        MLE of t copula using fixed P from empirical estimation;
        """
        P = args[0]
        U = args[1]
        T = U.shape[0]
        mle = 0
        for i in range(T):
            X = np.array([tdist.ppf(u, nu) for u in U[i, :]]).reshape(-1, 1)
            Y = np.sum([np.log(tdist.pdf(tdist.ppf(u, nu), nu)) for u in U[i, :]])
            mle += np.log(self._t_pdf(nu, P, X)) - Y
        return -mle
        
    def fitCopula(self, model='s', garch=(1, 1), arma=(1, 1), archpow=1, 
                  dist='norm', dist_dcc='norm', marginal='empirical'):
        """
        Fit a Garch-DCC-Copula model with t-copula;
        """
        assert (marginal in ['empirical', 'norm', 'std']), \
            "marginal should be 'empirical', 'norm' or 'std'!"
        
        # fit DCC
        self.fitDCC(model, garch, arma, archpow, dist, dist_dcc)
        
        # DCC residuals
        residuals = self.residuals(model='correlation').values  
        
        # fit marginals and estimate P
        U, paras = self._marginals(residuals, marginal)
        P = self._P(U)
        
        # max mle
        result = minimize(fun=self._mle_t_copula,
                          x0=np.array([10]),
                          args=(P, U),
                          method='SLSQP',
                          bounds=[(2, None)])
    
        # store results
        self._val = result.fun
        self._model['tail'] = 't-Copula'
        self._model['tail marginals'] = marginal
        self._paras['tail'] = {'nu': result.x[0], 'P': P}
        
        if marginal == 'empirical':
            self._paras['tail marginals'] = None
        elif marginal == 'norm':
            col = self._X.columns
            idx = ['mu', 'std']
            self._paras['tail marginals'] = pd.DataFrame(paras, index=idx, 
                                                         columns=col)
        else:
            col = self._X.columns
            idx = ['df', 'mu', 'std']
            self._paras['tail marginals'] = pd.DataFrame(paras, index=idx, 
                                                         columns=col)
        
        # warning if not converged
        if not result.success:
            print('\n')
            print('-'*50)
            print('Copula is not converged! \n')
            print('-'*50)
        
#%% Return Fields

    def model(self):
        """
        Print model specifics
        """
        print('\n')
        print('-' * 50)
        for key in self._model.keys():
            print(key+': '+self._model[key])
        print('-' * 50)

    def residuals(self, model='variance'):
        """
        Return residuals at each modeling step;
        """
        assert (model in self._paras.keys()), \
            "Unrecognized model name or model hasn't been fitted yet!"
        cols = self._X.columns
        idx = self._X.index

        if model == 'variance':
            residuals = self._residuals  # mean-variance corrected residuals
        elif model == 'mean':
            residuals = self._residuals*self._stds  # mean-corrected residuals
        else:
            n = self._X.shape[1]
            T = self._X.shape[0]
            a = self._residuals*self._stds  # mean-corrected residuals
            residuals = np.empty([T, n])
            for i in range(T):
                at = a[i, :].reshape(-1, 1)
                std = self._stds[i, :]
                Dt = self._Dt(std)
                R = self._corrs[:, :, i]
                H = self._Ht_sqrt_inv(Dt, R)
                residuals[i, :] = H.dot(at).flatten()  # mean-cov corrected residuals

        return pd.DataFrame(residuals, columns=cols, index=idx)
    
    def sigma(self):
        """
        Return fitted standard deviations
        """
        if 'variance' not in self._paras.keys():
            print('variance model has not been fitted yet!')
            return 
        cols = self._X.columns
        idx = self._X.index
        return pd.DataFrame(self._stds, columns=cols, index=idx)
    
    def corr(self):
        """
        Return fitted correlations
        """
        if 'correlation' not in self._paras.keys():
            print('correlation model has not been fitted yet!')
            return 
        return self._corrs

    def cov(self):
        """
        Return fitted covariance matrix
        """
        if 'correlation' not in self._paras.keys():
            print('correlation model has not been fitted yet!')
            return
        T = self._corrs.shape[2]
        d = self._corrs.shape[0]
        cov = np.empty([d, d, T])
        for i in range(T):
            corr = self._corrs[:, :, i]
            std = np.diag(self._stds[i, :])  # diagonal matrix of sigma
            cov[:, :, i] = std.dot(corr).dot(std)
        return cov

    def params(self, model='all'):
        """
        Return model parameters
        """
        assert (model in self._paras.keys() or model == 'all'), \
            "Unrecognized model name or model hasn't been fitted yet!"
        if model == 'all':
            return self._paras
        else:
            return self._paras[model]

    def ssigma(self, lambda_=0.94):
        """
        Return sample standard deviations;
        RiskMetrics proposes 0.94 for trading and 0.97 for investing;
        :param lambda_: exponential smooth parameter;
                        1 means usual equal-weighted sample std
        :return: pd.Series with sample std
        """
        assert (0 < lambda_ <= 1), "decay factor must be in (0, 1] range!"
        if lambda_ == 1:
            return self._X.std()
        else:
            T = self._X.shape[0]
            lam = np.array([lambda_**t for t in range(T)]).reshape(-1, 1)
            x = (self._X.copy() - self._X.mean())**2
            return np.sqrt((1-lambda_)*((lam*x).sum()))

    def scov(self, lambda_=0.94):
        """
        Return sample covariance matrix;
        RiskMetrics proposes 0.94 for trading and 0.97 for investing;
        :param lambda_: exponential smooth parameter;
                        1 means usual equal-weighted sample std
        :return: pd.DataFrame with sample covariance
        """
        assert (0 < lambda_ <= 1), "decay factor must be in (0, 1] range!"
        if lambda_ == 1:
            return self._X.cov()
        else:
            d = self._X.shape[1]
            T = self._X.shape[0]
            cols = self._X.columns
            lam = np.array([lambda_ ** t for t in range(T)]).reshape(-1, 1)
            x = (self._X.copy() - self._X.mean()).values  # mean-corrected returns

            cov = np.empty([d, d])
            for i in range(d):
                for j in range(d):
                    ri = x[:, i].reshape(-1, 1)
                    rj = x[:, j].reshape(-1, 1)
                    cov[i, j] = (1 - lambda_) * ((lam*ri).T.dot(rj))
            return pd.DataFrame(cov, columns=cols, index=cols)

    def scorr(self, lambda_=0.94):
        """
        Return sample correlation matrix;
        RiskMetrics proposes 0.94 for trading and 0.97 for investing;
        :param lambda_: exponential smooth parameter;
                        1 means usual equal-weighted sample std
        :return: pd.DataFrame with sample correlation
        """
        assert (0 < lambda_ <= 1), "decay factor must be in (0, 1] range!"
        if lambda_ == 1:
            return self._X.corr()
        else:
            cols = self._X.columns
            std = self.ssigma(lambda_).values
            cov = self.scov(lambda_).values
            corr = np.diag(1/std).dot(cov).dot(np.diag(1/std))
            return pd.DataFrame(corr, columns=cols, index=cols)

    #%% Forecast
    
    def _forecastidx(self, step):
        """
        Private function to generate forward index
        """
        # randomly sample 5 time index to find time delta
        index = self._X.index
        random = np.random.randint(0, len(index)-1, 5)
        deltas = [index[t+1]-index[t] for t in random]
        delta = np.min(deltas)

        # add proper time delta
        idx = [self._X.index[-1] + delta*i for i in range(1, step+1)]
        return idx
    
    def _recovergarch(self, col):
        """
        Recover GARCH model specifics
        """
        # variance model specifics
        v_model = self._model['variance'].split('(')
        v = v_model[0]
        gOrder = r.c(int(v_model[1][0]), int(v_model[1][3]))
        
        # mean model specifics
        m_model = self._model['mean']
        aOrder = r.c(int(m_model[5]), int(m_model[8]))
        if 'm' in m_model:
            am = True
            apow = int(m_model[-1])
        else:
            am = False
            apow = 1
        
        # distribution
        dist = self._model['variance distribution']
        
        # fixed parameters
        m_para = self._paras['mean'][col]
        v_para = self._paras['variance'][col]
        para = m_para.append(v_para)

        spec = r.ugarchspec(

                variance_model=r.list(model=v,
                                      garchOrder=gOrder
                                      ),

                mean_model=r.list(armaOrder=aOrder,
                                  archm=am,
                                  archpow=apow
                                  ),

                distribution_model=dist,
                
                fixed_pars=para
                )
        return spec
    
    def _forecast_std(self, step):
        """
        Mean and standard deviation forecast;
        """
        n = self._X.shape[1]
        cols = self._X.columns
        
        mean = np.empty([step, n])
        std = np.empty([step, n])
        
        for i in range(n):
            spec = self._recovergarch(cols[i])
            x = self._X[cols[i]]
            forecast = r.ugarchforecast(spec, data=x, n_ahead=step)
            mean[:, i] = r.fitted(forecast).flatten()
            std[:, i] = r.sigma(forecast).flatten()
        return mean, std
    
    def _forecast_corr(self, step):
        """
        Correlation forecast using method 2 in the paper;
        """
        a = self._paras['correlation']['a']
        b = self._paras['correlation']['b']
        
        Q_bar = self._Q_bar()
        R_bar = self._Rt(Q_bar)
        R_expected = self._Rt(self._Q)
        
        n = self._X.shape[1]
        corr = np.empty([n, n, step])
        for i in range(step):
            corr[:, :, i] = (1-(a+b)**i)*R_bar + (a+b)**i*R_expected
            
        return corr
    
    def _forecast_cov(self, step):
        """
        Covariance matrix forecast;
        """
        n = self._X.shape[1]
        
        _, std = self._forecast_std(step)
        corr = self._forecast_corr(step)
        
        cov = np.empty([n, n, step])
        for i in range(step):
            D = self._Dt(std[i, :])
            R = corr[:, :, i]
            cov[:, :, i] = D.dot(R).dot(D)
        return cov
    
    def forecast(self, step=10, target='std'):
        """
        Return forecast mean, standard deviation or correlations by given steps;
        DCC forecast uses method 2 in the paper;
        """
        assert (target in ['mean', 'std', 'corr', 'cov']), \
            "target should be one of the 'mean', 'std', 'corr' or 'cov'!"
        
        # map to check if target available
        maps = {'mean': 'mean', 'std': 'variance', 
                'corr': 'correlation', 'cov': 'correlation'}    
        assert (maps[target] in self._paras.keys()), \
            "model hasn't been fitted yet!"
        
        # mean or standard deviation forecast
        if target in ['mean', 'std']:   
            mean, std = self._forecast_std(step)
            
            # output if mean or std
            cols = self._X.columns
            idx = self._forecastidx(step)
            if target == 'mean':
                return pd.DataFrame(mean, columns=cols, index=idx)
            elif target == 'std':
                return pd.DataFrame(std, columns=cols, index=idx)
        
        # correlation forecast        
        elif target == 'corr':
            return self._forecast_corr(step)
        
        # covariance forecast
        else:
            return self._forecast_cov(step)

#%% Sampling
        
    def _sample_copula(self, n):
        """
        Simulating n samples from the t copula;
        """
        d = self._X.shape[1]  # dimensions of data
        
        # simulate multivariate normal variables
        P = self._paras['tail']['P']
        mu = np.zeros(d)
        mn = np.random.multivariate_normal(mu, P, n)  # each row is a sample
        
        # simulate Chi-squared variables
        nu = self._paras['tail']['nu']
        xi = np.random.chisquare(nu, (n, 1))
        
        # calculate t-copula variables           
        U = tdist.cdf(mn/np.sqrt(xi/nu), nu)  
        return U
        
    def _reverse_marginal(self, U):
        """
        Transforming t-copula variables into DCC residuals by marginals:
        """
        n = U.shape[0]
        d = U.shape[1]
        
        # empirical quantile
        if self._model['tail marginals'] == 'empirical':
            residuals = self.residuals(model='correlation').values
            Y = np.empty([n, d])  # container of transformed variables
            for i in range(d):
                Y[:, i] = np.quantile(residuals[:, i], U[:, i])
            return Y
        
        # normal quantile
        elif self._model['tail marginals'] == 'norm':
            Y = np.empty([n, d])  # container of transformed variables
            for i in range(d):
                mu = self._paras['tail marginals'].values[0, i]
                std = self._paras['tail marginals'].values[1, i]
                Y[:, i] = norm.ppf(U[:, i], mu, std)
            return Y
        
        # t quantile
        else:
            Y = np.empty([n, d])  # container of transformed variables
            for i in range(d):
                df = self._paras['tail marginals'].values[0, i]
                mu = self._paras['tail marginals'].values[1, i]
                std = self._paras['tail marginals'].values[2, i]
                Y[:, i] = tdist.ppf(U[:, i], df, mu, std)
            return Y
            
    def _sample_dcc(self, n):
        """
        Sample DCC residuals;
        """
        d = self._X.shape[1]
        
        # multivariate normal
        if self._model['correlation distribution'] == 'norm':
            P = np.eye(d)
            mu = np.zeros(d)
            return np.random.multivariate_normal(mu, P, n)
        
        # multivariate t
        elif self._model['correlation distribution'] == 'std':
            P = np.eye(d)
            mu = np.zeros(d)
            mn = np.random.multivariate_normal(mu, P, n)  # multivariate normal
            nu = self._paras['correlation']['nu']
            xi = np.random.chisquare(nu, (n, 1))  # Chisquared variables
            return mn/np.sqrt(xi/nu)
        
        # multivariate skew t
        else:
            nu = self._paras['correlation']['nu']
            sig = self._paras['correlation'].values[-d:].reshape(-1, 1)
            omega = self._omega(nu, sig)
            D = self._D(omega)
            delta = self._delta(sig, D)
            xi = self._xi(nu, sig, omega)
            rmst = r('sn::rmst')  # request sn package in R
            return rmst(n, xi, omega, delta, nu)
    
    def _reverse_dcc(self, Z):
        """
        Transforming DCC residuals into final samples by 
        cov matrix and mean vector;
        Z should be n*d*step array;
        return samples are n*d*step array;
        """
        n = Z.shape[0]  # number of samples
        d = Z.shape[1]  # number of variables
        step = Z.shape[2]  # number of time step forecasted
        
        cov = self.forecast(step, target='cov')  # forecast covariance matrix
        mu = self.forecast(step, target='mean').values  # forecast mean vector
        sample = np.empty([n, d, step])
        
        for i in range(step):
            H = np.linalg.cholesky(cov[:, :, i])  # Cholesky factorization 
            sample[:, :, i] = Z[:, :, i].dot(H.T) + mu[i:i+1, :]
            
        return sample
    
    def _sample_garch(self, n):
        """
        Sample GARCH residuals
        """
        d = self._X.shape[1]
        
        # normal 
        if self._model['variance distribution'] == 'norm':
            Z = np.random.normal(size=[n, d])
            
        # student t
        elif self._model['variance distribution'] == 'std':
            Z = np.empty([n, d])
            for i in range(d):
                nu = self._paras['variance']['shape']
                Z[:, i] = np.random.standard_t(nu, n)
        
        # skew norm        
        elif self._model['variance distribution'] == 'snorm':
            Z = np.empty([n, d])
            for i in range(d):
                a = self._paras['variance']['skew']
                rmsn = r('sn::rmsn')
                Z[:, i] = rmsn(n, Omega=1, alpha=a).flatten()
                
        # skew t
        else:
            Z = np.empty([n, d])
            for i in range(d):
                a = self._paras['variance']['skew']
                nu = self._paras['variance']['shape']
                rmst = r('sn::rmst')
                Z[:, i] = np.array(rmst(n, Omega=1, alpha=a, nu=nu)).flatten()
    
        return Z
    
    def _reverse_garch(self, Z):
        """
        Transforming GARCH residuals into final samples;
        """
        n = Z.shape[0]  # number of samples
        d = Z.shape[1]  # number of variables
        step = Z.shape[2]  # number of time step forecasted
        
        std = self.forecast(step, target='std').values  # forecast variance
        mu = self.forecast(step, target='mean').values  # forecast mean vector
        sample = np.empty([n, d, step])
        
        for i in range(step):
            sample[:, :, i] = Z[:, :, i]*std[i:i+1, :] + mu[i:i+1, :]
            
        return sample
    
    def sample(self, step=10, n=100):
        """
        Making n samples of given step by all models available;
        return samples are n*d*step array;
        """
        if 'variance' not in self._model.keys():
            print("No model has been fitted!")
            return
        
        d = self._X.shape[1] 
        
        # model residuals
        Z = np.empty([n, d, step]) 
        
        # sample from correlation model level
        if 'correlation' in self._model.keys():
            
            if 'tail' in self._model.keys():
                for i in range(step):
                    U = self._sample_copula(n)  # t-copula samples
                    Z[:, :, i] = self._reverse_marginal(U)  
            else:
                for i in range(step):
                    Z[:, :, i] = self._sample_dcc(n)
            
            # final samples
            samples = self._reverse_dcc(Z)
            return samples
            
        # sample from variance model level
        else:
            for i in range(d):
                Z[:, :, i] = self._sample_garch(n)
                
            # final samples
            samples = self._reverse_garch(Z)
            return samples
