#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:46:51 2019

@author: victor
"""

#%% import libs

from pandas import Series, DataFrame
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import MinCovDet


#%% Risk Metrics

class RiskMetrics(object):
    """
    A tool to calculate risk metrics;
    Data input should be DataFrame with columns being assets;
    frequency is number of periods per year (e.g. 252 for daily data);
    If input is return it should be simple periodic return;
    """
    
    def __init__(self, X, XType='return', frequency=12):
        # check inputs
        assert(isinstance(X, DataFrame)), \
            'Please use DataFrame for X!'
        
        assert(XType in ['return', 'price']), \
            "bType should be 'return' or 'price'!"
             
        if XType == 'price':
            X = (X/X.shift(1)-1).drop(X.index[0])
        
        # initial inputs
        self._X = X
        self._f = frequency

#%% Asset-specific Risk: Basic
    
    def returns(self, annualize=True):
        if annualize:
            return self._X.mean() * self._f
        else:
            return self._X.mean()
        
    def volatility(self, annualize=True):
        if annualize:
            return self._X.std() * np.sqrt(self._f)
        else:
            return self._X.std()

    def skewness(self):
        return self._X.skew()

    def kurtosis(self):
        """
        Excess kurtosis
        """
        return self._X.kurt()

    def covariance(self, annualize=True):
        if annualize:
            return self._f * self._X.cov()
        else:
            return self._X.cov()
        
    def correlation(self):
        return self._X.corr()
    
    def srho(self):
        """
        Spearman's Rho
        """
        col = self._X.columns
        r = stats.spearmanr(self._X)[0]
        return DataFrame(r, columns=col, index=col)

    def ktau(self):
        """
        Kendall's tau
        """
        col = self._X.columns
        d = self._X.shape[1]
        matrix = np.empty([d, d])
        
        # compute pairwise tau
        for i in range(d-1):
            for j in range(i+1, d):
                matrix[i, j] = stats.kendalltau(self._X.iloc[:, i].values,
                                                self._X.iloc[:, j].values)[0]
        
        # fill the matrix
        for i in range(1, d):
            for j in range(i):
                matrix[i, j] = matrix[j, i]
        for i in range(d):
            matrix[i, i] = 1
        
        return DataFrame(matrix, columns=col, index=col)
    
    def drawdowns(self):
        cumu = (self._X + 1).cumprod()-1
        drawdowns = DataFrame(index=cumu.index, columns=cumu.columns)
        m = drawdowns.shape[0]
        n = drawdowns.shape[1]
        for i in range(n):
            series = cumu.iloc[:, i].values
            for j in range(m):
                hwm = max(max(series[:j+1]), 0)  # high water mark
                d = (series[j]-hwm)/(1+hwm)  # current drawdowns
                drawdowns.iloc[j, i] = d
        return drawdowns

    def maxDrawdown(self):
        drawdowns = self.drawdowns()
        return np.min(drawdowns)
    
    def sharpe(self, riskfree=0.02):
        ret = self._X.mean() * self._f
        vol = self._X.std() * np.sqrt(self._f)
        return (ret - riskfree) / vol

    def adjSharpe(self, riskfree=0.02):
        sharpe = self.sharpe(riskfree)
        skew = self._X.skew()
        kurt = self._X.kurtosis()
        return sharpe * (1 + skew/6*sharpe - kurt/24*sharpe**2)

    def _checkBenchmark(self, benchmark, bType):
        assert(bType in ['return', 'price']), \
            "bType should be 'return' or 'price'!"
        assert(isinstance(benchmark, DataFrame)), \
            "Please use DataFrame type fro benchmark!"
        
        if bType == 'price':
            benchmark = (benchmark/benchmark.shift(1)-1).drop(benchmark.index[0])

        assert(benchmark.shape[0] == self._X.shape[0]),\
            "Benchmark should have same length as input data!"
        return benchmark
    
    def infoRatio(self, benchmark, bType='return'):
        benchmark = self._checkBenchmark(benchmark, bType)
        n = self._X.shape[1]
        m = benchmark.shape[1]
        data = self._X.values
        IR = np.empty([n, 1])
        for i in range(m):
            trackerror = data - benchmark.values[:, i].reshape(-1, 1)
            excess = trackerror.mean(axis=0) * self._f
            vol = trackerror.std(axis=0) * np.sqrt(self._f)
            ir = excess / vol
            IR = np.concatenate([IR, ir.reshape(-1, 1)], axis=1)
        IR = DataFrame(IR[:, 1:],
                       index=self._X.columns,
                       columns=benchmark.columns)
        return IR
    
    def sortino(self, MAR=0, method='gaussian'):
        """
        Discrete method is fundamentally wrong, but widely used;
        Gaussian solution is from paper 'An Objective Analysis of Alternative 
        Risk-to-Reward Ratios';
        """
        
        assert(method in ['discrete', 'gaussian']), \
            "method should be 'discrete' or 'gaussian'!"
        if method == 'discrete':
            excess = self._X - MAR/self._f
            excess[excess > 0] = 0
            dd = np.sqrt((excess**2).sum() / excess.shape[0])
            return (self._X.mean()*self._f - MAR) / (dd*np.sqrt(self._f))
        else:
            ret = self._X.mean() * self._f
            vol = self._X.std() * np.sqrt(self._f)
            dd = (vol**2 + (ret-MAR)**2) * stats.norm.cdf(-(ret-MAR)/vol) - \
                 (ret-MAR)*vol * stats.norm.pdf((ret-MAR)/vol)
            return (ret - MAR) / np.sqrt(dd)

    def winRate(self):
        """
        The positive % periods
        """
        return np.sum(self._X > 0, axis=0)/self._X.shape[0]

    def linearReg(self, benchmark, bType='return'):
        """
        Calculate alpha, beta, tstats and Rsquare wrt benchmarks
        Multi benchmarks supported
        """
        
        benchmark = self._checkBenchmark(benchmark, bType)
        Y = self._X.values
        x = np.concatenate((np.ones([66, 1]), benchmark.values), axis=1)
        m = x.shape[1]
        summary = np.empty([1, 2*m+1])
        
        for i in range(Y.shape[1]):
            
            y = Y[:, i].reshape(-1, 1)
            # betas
            betas = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
            
            e = y-(x.dot(betas))  # errors
            
            ssr = e.T.dot(e)  # sum of squared error
            sst = (y-y.mean()).T.dot((y-y.mean()))
            rsquared = 1 - ssr/sst
            
            n = x.shape[0]
            k = x.shape[1]
            varhat = e.T.dot(e)/(n-k)  # error variance
            betacov = varhat*np.linalg.inv(x.T.dot(x))  # beta cov matrix
            tstats = betas/np.sqrt(np.diagonal(betacov).reshape(-1, 1))
            
            s = np.concatenate([betas.T, tstats.T, rsquared], axis=1)
            summary = np.concatenate([summary, s])
        
        index = self._X.columns
        columns = ['alpha'] + [f'beta{i}' for i in range(1, m)] + \
                  ['t_alpha'] + [f't_beta{i}' for i in range(1, m)] + \
                  ['Rsquared']    
        summary = DataFrame(summary[1:, :], columns=columns, index=index)
        return summary

#%% Asset-specific Risk: VaR & CVaR

    def var(self, alpha=0.05, method='MCF', cleaning=False):
        """
        Value-at-Risk;
        Historical method is simply ranking data and picking the threshold;
        Gaussian method is calculated from normal distribution cdf;
        MCF stands for Modified Cornish-Fisher VaR, see more details in 
        'Estimation and decomposition of downside risk for portfolios with 
        non-normal returns';
        Cleaning will replace outliers and produce a more robust estimation,
        if a more conservative approach keeping all original data is preferred,
        cleaning should be switch to False;
        'If your data is fat-tailed and/or skewed enough to cause the breakdown 
        in mVaR and mES, or if you want to estimate the very extreme downside 
        risk (α = 0.01 − 0.001 or smaller), a copula-based approach as proposed 
        by Embrechts et al (2001) may be a more appropriate estimator of VaR
        and ES.';
        """
        
        assert(method in ['empirical', 'gaussian', 'MCF']), \
            "method should be 'empirical' or 'gaussian' or 'MCF'!"
        
        data = self._X.copy()
        if cleaning:
            data = self.dataCleaning(method='univariate')
        
        if method == 'empirical':
            n = data.shape[1]  # number of assets
            m = data.shape[0]  # number of periods
            mini = np.ceil(1/alpha)  # minimum recommanded number of periods
            if m < mini:
                print("\n")
                print(f"The length of data is smaller than {mini}!")
            t = int(np.ceil(m * alpha)-1)  # threshold
            VaR = []
            for i in range(n):
                VaR.append(data.iloc[:, i].sort_values()[t])
            VaR = Series(VaR, index=data.columns)
            return -VaR  # VaR is defined as positive number
        
        elif method == 'gaussian':
            mu = data.mean()
            sig = data.std()
            z = stats.norm.ppf(alpha)  # inversed pdf to find threshold at alpha
            VaR = -mu - sig * z
            return VaR
            
        else:
            mu = data.mean()
            sig = data.std()
            skew = data.skew()
            kurt = data.kurt()
            z = stats.norm.ppf(alpha)  # inverse pdf to find threshold at alpha
            gau = -mu - sig * z  # gaussian VaR
            VaR = gau + sig * (-(z**2-1)/6*skew - (z**3-3*z)/24*kurt +
                                (2*z**3-5*z)/36*skew**2)
            return VaR

    def es(self, alpha=0.05, method='MCF', cleaning=False):
        """
        Expected shortfall (CVaR);
        Historical method is simply ranking data and averaging the returns 
        below threshold;
        Gaussian method is calculated from normal distribution cdf;
        MCF stands for Modified Cornish-Fisher CVaR, see more details in 
        'Estimation and decomposition of downside risk for portfolios with 
        non-normal returns';
        Cleaning will replace outliers and produce a more robust estimation,
        if a more conservative approach keeping all original data is preferred,
        cleaning should be switch to False;
        'If your data is fat-tailed and/or skewed enough to cause the breakdown 
        in mVaR and mES, or if you want to estimate the very extreme downside 
        risk (α = 0.01 − 0.001 or smaller), a copula-based approach as proposed 
        by Embrechts et al (2001) may be a more appropriate estimator of VaR
        and ES.';
        """
        assert(method in ['empirical', 'gaussian', 'MCF']), \
            "method should be 'empirical' or 'gaussian' or 'MCF'!"
        
        data = self._X.copy()
        if cleaning:
            data = self.dataCleaning(method='univariate')
        
        if method == 'empirical':
            n = data.shape[1]  # number of assets
            m = data.shape[0]  # number of periods
            mini = np.ceil(1/alpha)  # minimum recommanded number of periods
            if m < mini:
                print("\n")
                print(f"The length of data is smaller than {mini}!")
            t = int(np.ceil(m * alpha)-1)  # threshold
            ES = []
            for i in range(n):
                ES.append(data.iloc[:, i].sort_values()[:t+1].mean())
            ES = Series(ES, index=data.columns)
            return -ES  # expected shortfall is defined as positive number
        
        elif method == 'gaussian':
            mu = data.mean()
            sig = data.std()
            z = stats.norm.ppf(alpha)
            ES = -mu + sig / alpha * stats.norm.pdf(z)
            return ES
        
        else:
            mu = data.mean()
            sig = data.std()
            skew = data.skew()
            kurt = data.kurt()
            z = stats.norm.ppf(alpha)  # inversed pdf to find threshold at alpha
            
            # 2nd order Cornish–Fisher expansion of the normal quantile function
            g = z + (z**2-1)/6*skew + (z**3-3*z)/24*kurt - \
                (2*z**3-5*z)/36*skew**2 
                
            def I(q):
                # q even 
                if q % 2 == 0:
                    J = lambda x: np.prod([2*j for j in np.arange(1, x+1)])
                    s = 0
                    for i in np.arange(1, q/2+1):
                        s += J(q/2) / J(i) * g**(2*i) * stats.norm.pdf(g)
                    return s + J(q/2) * stats.norm.pdf(g)
                # q odd
                else:
                    J = lambda x: np.prod([2*j+1 for j in np.arange(0, x+1)])
                    qstar = (q-1)/2
                    s = 0
                    for i in np.arange(0,qstar+1):
                        s += J(qstar) / J(i) * g**(2*i+1) * stats.norm.pdf(g)
                    return s - J(qstar) * stats.norm.cdf(g)
               
            E = -1/alpha * ( 
                stats.norm.pdf(g) + 
                (I(4) - 6*I(2) + 3*stats.norm.pdf(g))/24*kurt +    
                (I(3) - 3*I(1))/6*skew +
                (I(6) - 15*I(4) + 45*I(2) - 15*stats.norm.pdf(g))/72*skew**2
                )
            # modify E to min(E,g) in case alpha is too small and E too big 
            E[E > g] = g[E > g]
            ES = -mu - sig * E
            return ES

#%% Asset-specific Risk: Tail Dependency

    @staticmethod
    def _empirical_cdf(X, Y, i, j):
        """
        Bivariate empirical copula cdf with parameter i, j;
        """
        # sort series
        Xs = np.sort(X)
        Ys = np.sort(Y)
        
        N = X.shape[0]
        c = 0
        for n in range(N):
            c += X[n] <= Xs[i] and Y[n] <= Ys[j]
        return c/N

    @staticmethod
    def _gumbel_cdf(u, v, theta):
        """
        Bivariate Gumbel copula cdf with parameter theta:
        C(u,v) := exp( −( (−lnu)^θ + (−lnv)^θ )^(1/θ) )
        """
        return np.exp(
                       -(
                         (-np.log(u))**theta + (-np.log(v))**theta
                        )**(1/theta)
                      )
    
    def _gumbel_pdf(self, u, v, theta):
        """
        Bivariate Gumbel copula pdf with parameter theta:
        c(U,V) = \frac{\partial^2 C(u,v)}{\partial v \partial u} =
                \frac{C(u,v)}{uv} \frac{((-\ln u)^{\theta} + 
                (-\ln v)^{\theta})^{\frac{2}
                {\theta} - 2 }}{(\ln u \ln v)^{1 - \theta}} 
                ( 1 + (\theta-1) \big((-\ln u)^\theta
                + (-\ln v)^\theta\big)^{-1/\theta})
        """
        cdf = self._gumbel_cdf(u, v, theta)
        a = ((-np.log(u))**theta + (-np.log(v))**theta)**(2/theta - 2)
        b = (np.log(u) * np.log(v))**(theta-1)
        c = 1 + (theta-1) * ( 
                ((-np.log(u))**theta + (-np.log(v))**theta)**(-1/theta)
                            )
        return cdf/u/v * a * b * c

    @staticmethod
    def _marginals(data, marginal):
        """
        Estimation of marginal cumulated density of each variable;
        """    
        d = data.shape[1]  # number of assets
        n = data.shape[0]  # number of periods
        cd = np.empty([n, d])
        
        if marginal == 'empirical':
            for i in range(d):
                for j in range(n):
                    cd[j, i] = np.sum(data[:, i] <= data[j, i]) / (n+1)
            return cd    
        
        elif marginal == 'gaussian':
            for i in range(d):
                mu, std = stats.norm.fit(data[:, i])
                for j in range(n):
                    cd[j, i] = stats.norm.cdf(data[j, i], loc=mu, scale=std)
            return cd
        
        elif marginal == 't':
            for i in range(d):
                df, mu, std = stats.t.fit(data[:, i])
                for j in range(n):
                    cd[j, i] = stats.t.cdf(data[j, i], df=df, loc=mu, scale=std)
            return cd
        
        else:
            for i in range(d):
                mu, std = stats.norm.fit(data[:, i])
                skew = stats.skew(data[:, i])
                kurt = stats.kurtosis(data[:, i])
                z = (data[:, i]-mu)/std  # standardize data
                P1 = lambda z: (z**2-1)*skew/6
                P2 = lambda z: (z**3-3*z)*kurt/24 + \
                               (z**5-10*z**3+15*z)*skew**2/72
                for j in range(n):
                    x = stats.norm.cdf(z[j]) - stats.norm.pdf(z[j]) * (
                              P1(z[j]) + P2(z[j]))
                    if x > 1:  # adjusted if cumulative prob larger than 1
                        x = 0.99999
                    cd[j, i] = x
            return cd
    
    def _gumble_mle(self, theta, *variables):
        """
        Maximum log-likelihood function of bivariate Gumbel copula;
        Negative sign for minimization purpose;
        """
        U = variables[0]
        V = variables[1]
        n = U.shape[0]
        ml = 0
        for i in range(n):
            ml += np.log(self._gumbel_pdf(U[i], V[i], theta))
        return -ml
    
    def _td(self, U, V, idx1, idx2, copula, start, method):
        """
        Bivariate tail dependency coefficient;
        Gumbel bivariate copula: lambda_= 2 - 2**(1/theta);
        Empirical copula: lambda_ is secant-based, slope-based, mixture-based;
        """
        
        if copula == 'gumbel':
            bnds = [(1, None)]
            if not start:
                start = np.random.randint(5, 20)  # random starting value
            x0 = np.array([start]) 
            opt = minimize(self._gumble_mle, x0, args=(U, V),
                           method='SLSQP', bounds=bnds)
            if not opt.success:
                print('\n')
                print('-'*50)
                print(f"MLE is not converged at {idx1}th & {idx2}th pair!")
                print('-'*50)
            theta = opt.x[0]
            return 2 - 2**(1/theta)
        
        else:
            N = U.shape[0]
            k = int(np.sqrt(N)) 
            if method == 'secant':
                C = self._empirical_cdf(U, V, k, k)
                return (N/k)*C
            elif method == 'slope':
                s1 = 0
                s2 = 0
                for i in range(k):
                    s1 += (i/N)**2
                    s2 += i/N*self._empirical_cdf(U, V, i, i)
                return s2/s1
            else:
                s1 = 0
                s2 = 0
                for i in range(k):
                    c = self._empirical_cdf(U, V, i, i)
                    s1 += (c-(i/N)**2) * (i/N-(i/N)**2)
                    s2 += (i/N-(i/N)**2)**2
                return s1/s2

    def tdm(self, copula='gumbel', marginal='empirical',
            start=None, method='slope'):
        """
        Bivariate tail dependency coefficient matrix;
        Gumbel bivariate copula;
        Parameter is estimated by pesudo-MLE;
        Returns are multiplied by -1 as Gumbel has right tail dependency;
        """
        
        assert(copula in ['empirical', 'gumbel']), \
            "copula should be 'empirical' or 'gumbel'!"
        assert(method in ['mixture', 'secant', 'slope']), \
            "method should be 'mixture','secant' or 'slope'!"
        assert(marginal in ['empirical', 'gaussian', 't', 'MCF']), \
            "marginal method should be 'empirical', 'gaussian', 't' or 'MCF'!"
        assert(start is None or start >= 1), \
            "start value must be None or not smaller than 1!"
        
        cols = self._X.columns
        
        if copula == 'gumbel':
            data = -self._X.copy().values
            
            d = data.shape[1]
            matrix = np.empty([d, d])
            
            # compute cumulated density
            cd = self._marginals(data, marginal)
            
            # compute pairwise tail dependency
            for i in range(d-1):
                for j in range(i+1, d):
                    matrix[i, j] = self._td(cd[:, i], cd[:, j], i+1, j+1,
                                            copula, start, method)
                            
        else:
            data = self._X.copy().values
            
            d = data.shape[1]
            matrix = np.empty([d, d])
            
            # compute pairwise tail dependency
            for i in range(d-1):
                for j in range(i+1, d):
                    matrix[i, j] = self._td(data[:, i], data[:, j], i+1, j+1,
                                            copula, start, method)
            
        # fill the matrix
        for i in range(1, d):
            for j in range(i):
                matrix[i, j] = matrix[j, i]
        for i in range(d):
            matrix[i, i] = 1
            
        return DataFrame(matrix, columns=cols, index=cols)
    
    def tdb(self, benchmark, bType='return',
            marginal='empirical', start=None):
        """
        Bivariate tail dependency coefficient with benchmarks;
        Gumbel bivariate copula;
        Parameter is estimated by pesudo-MLE;
        Returns are multiplied by -1 as Gumbel has right tail dependency;
        """
        
        assert(marginal in ['empirical', 'gaussian', 't', 'MCF']), \
            "marginal method should be 'empirical', 'gaussian', 't' or 'MCF'!"
        assert(start is None or start >= 1), \
            "start value must be None or not smaller than 1!"
        
        benchmark = self._checkBenchmark(benchmark, bType)
        
        idx = self._X.columns
        col = benchmark.columns
        data = -self._X.copy().values
        benchmark = -benchmark.values
        
        d = data.shape[1]
        b = benchmark.shape[1]
        matrix = np.empty([d, b])
        
        # compute cumulated density
        cd_d = self._marginals(data, marginal)
        cd_b = self._marginals(benchmark, marginal)
        
        # compute pairwise tail dependency
        for i in range(d):
            for j in range(b):
                matrix[i, j] = self._td(cd_d[:, i], cd_b[:, j], i+1, j+1, start)
        
        return DataFrame(matrix, columns=col, index=idx)

#%% Portfolio Risk: Basic

    def coskewness(self):
        """
        Co-skewness matrix;
        n * n**2;
        Follow same paper as ES and VaR
        """
        
        r = self._X.values 
        mu = np.mean(r, axis=0).reshape(1, -1)
        x = r - mu
        n = x.shape[1]  # number of assets
        m = x.shape[0]
        M3 = []
        for i in range(n):
            s = np.empty([n, n])  # S block inside Kronecker product
            for j in range(n):
                for k in range(n):
                    # E[(r_i - mu_i)*(r_j - mu_j)*(r_k - mu_k)]
                    s[j, k] = np.sum(x[:, i] * x[:, j] * x[:, k]) / (m-1)
            M3.append(s)
        M3 = np.concatenate(M3, axis=1)
        return M3

    def cokurtosis(self):
        """
        Co-kurtosis matrix
        n * n**3
        """
        r = self._X.values 
        mu = np.mean(r, axis=0).reshape(1, -1)
        x = r - mu
        n = x.shape[1]  # number of assets
        m = x.shape[0]
        M4 = []
        for i in range(n):
            for j in range(n):
                K = np.empty([n, n])  # k block inside Kronecker product
                for k in range(n):
                    for l in range(n):
                        # E[(r_i - mu_i)*(r_j - mu_j)*(r_k - mu_k)*(r_l - mu_l)]
                        K[k, l] = np.sum(x[:, i]*x[:, j]*x[:, k]*x[:, l])/(m-1)
                M4.append(K)
        M4 = np.concatenate(M4, axis=1)
        return M4

    def dataCleaning(self, alpha=0.05, method='multivariate'):
        """
        Covariance matrix estimated by Minimum Covariance Determinant (MCD);
        Cleaning ouliers based on same paper as VaR;
        'the primary value of data cleaning lies in creating a more robust and 
        stable estimation of the distribution generating the large majority of 
        the return data. The increased robustness and stability of the 
        estimated moments utilizing cleaned data should be used for portfolio 
        construction. If a portfolio manager wishes to have a more conservative 
        risk estimate, cleaning may not be indicated for risk monitoring.';
        Single variate method treat each column independently while the
        multivariate calculate covariance matrix and are better for portfolio;
        """
        
        assert(method in ['univariate', 'multivariate']), \
            "method should be 'univariate' or 'multivariate'!"
        
        data = self._X.copy()
        
        if method == 'multivariate':
            X = data.values
            T = X.shape[0]   
            n = X.shape[1] 
            
            # estimate mu and cov by MCD with (1-alpha)T rows
            MCD = MinCovDet(support_fraction=1-alpha).fit(X)
            cov = MCD.covariance_
            mu = X[MCD.support_, :].mean(axis=0).reshape(1, -1)

            # calculate Mahalanobis distance
            xmu = X-mu
            invcov = np.linalg.inv(cov)    
            mahad = []
            for i in range(T):
                x = xmu[i, :].reshape(-1, 1)
                mahad.append(x.T.dot(invcov).dot(x)[0, 0])
            mahad = np.array(mahad)
            
            # find threshold of Maha distance and Chi-squared dist.    
            t = int(T*(1-alpha))    
            thres_maha = np.sort(mahad)[t-1]  # threshold for distance
            thres_chi = stats.chi2.ppf(0.999, n)  # threshold for chi2 dist
            thres_max = max(thres_maha, thres_chi)
               
            # identify and correct outlier
            idx = np.all([mahad > thres_maha, mahad > thres_chi], axis=0)
            if np.any([idx]):
                X[idx, :] *= np.sqrt(thres_max / mahad[idx].reshape(-1, 1))
        
        else:
            T = data.shape[0]   
            n = 1
            container = np.empty([T, 1])
            for i in range(self._X.shape[1]):
                
                X = data.values[:, i].reshape(-1, 1)
                
                # estimate mu and cov by MCD with (1-alpha)T rows
                MCD = MinCovDet(support_fraction=1-alpha).fit(X)
                cov = MCD.covariance_
                mu = X[MCD.support_, :].mean(axis=0).reshape(1, -1)
                
                # calculate Mahalanobis distance
                xmu = X-mu
                invcov = np.linalg.inv(cov)    
                mahad = []
                for j in range(T):
                    x = xmu[j, :].reshape(-1, 1)
                    mahad.append(x.T.dot(invcov).dot(x)[0, 0])
                mahad = np.array(mahad)
                
                # find threshold of Maha distance and Chi-squared dist.    
                t = int(T*(1-alpha))    
                thres_maha = np.sort(mahad)[t-1]  # threshold for distance
                thres_chi = stats.chi2.ppf(0.999, n)  # threshold for chi2 dist
                thres_max = max(thres_maha, thres_chi)
                   
                # identify and correct outlier
                idx = np.all([mahad > thres_maha, mahad > thres_chi], axis=0)
                if np.any([idx]):
                    X[idx, :] *= np.sqrt(thres_max / mahad[idx].reshape(-1, 1))
                container = np.concatenate((container, X), axis=1)
            X = container[:, 1:]
    
        cleaned = DataFrame(X, columns=self._X.columns, index=self._X.index)
        return cleaned

#%% Portfolio Risk: VaR & ES
    
    def _checkWeights(self, weights):
        """
        private function to check weight input
        """
        assert(any([
                   isinstance(weights, list),
                   isinstance(weights, tuple),
                   isinstance(weights, np.ndarray)
                   ])
               ), "weights should be list, tuple or np.array!"
                
        if isinstance(weights, list) or isinstance(weights, tuple):
            assert(len(weights) == self._X.shape[1]), \
                "weights should have same lengths as input data columns!"
            w = np.array(weights).reshape(-1, 1)
        else:
            assert(weights.shape[0] == self._X.shape[1]), \
                "weights array should have rows as input data columns!"
            if weights.ndim == 1:
                w = weights.reshape(-1, 1)
            else:
                assert(weights.shape[1] == 1), \
                    "weights array should have one column!"
                w = weights
        return w

    def var_p(self, weights, alpha=0.05, method='MCF', cleaning=False):
        """
        Value-at-Risk for a portfolio;
        Create this for completeness;
        Highly recommand calculate portfolio returns first and use VaR method 
        above directly;
        """
        
        assert(method in ['gaussian', 'MCF']), \
            "method should be 'gaussian' or 'MCF'!"
        
        w = self._checkWeights(weights)
            
        data = self._X.copy()
        if cleaning:
            data = self.dataCleaning(method='multivariate')
            
        if method == 'gaussian':
            mu = data.mean()
            cov = data.cov()
            m2 = w.T.dot(cov).dot(w)
            z = stats.norm.ppf(alpha)  # inversed pdf to find threshold at alpha
            VaR = -w.T.dot(mu) - np.sqrt(m2) * z
            return VaR[0, 0]
            
        else:
            mu = data.mean()
            cov = data.cov()
            M3 = self.coskewness()
            M4 = self.cokurtosis()
            m2 = w.T.dot(cov).dot(w)  # the qth centered portfolio moment
            m3 = w.T.dot(M3).dot(np.kron(w, w))
            m4 = w.T.dot(M4).dot(np.kron(np.kron(w, w), w))
            skew = m3/m2**1.5
            kurt = m4/m2**2 - 3
            z = stats.norm.ppf(alpha)  # inversed pdf to find threshold at alpha
            gau = -w.T.dot(mu) - np.sqrt(m2) * z  # gaussian VaR
            VaR = gau + np.sqrt(m2) * (-(z**2-1)/6*skew - (z**3-3*z)/24*kurt +
                                        (2*z**3-5*z)/36*skew**2)
            return VaR[0, 0]
    
    def es_p(self, weights, alpha=0.05, method='MCF', cleaning=False):
        """
        CVaR for a portfolio;
        Create this for completeness;
        Highly recommand calculate portfolio returns first and use ES method 
        above directly;
        """
        
        assert(method in ['gaussian', 'MCF']), \
            "method should be 'gaussian' or 'MCF'!"
        
        w = self._checkWeights(weights)
            
        data = self._X.copy()
        if cleaning:
            data = self.dataCleaning(method='multivariate')
        
        if method == 'gaussian':
            mu = data.mean()
            cov = data.cov()
            m2 = w.T.dot(cov).dot(w)
            z = stats.norm.ppf(alpha)  # inversed pdf to find threshold at alpha
            ES = -w.T.dot(mu) + np.sqrt(m2) / alpha * stats.norm.pdf(z)
            return ES[0, 0]
        
        else:
            mu = data.mean()
            cov = data.cov()
            M3 = self.coskewness()
            M4 = self.cokurtosis()
            m2 = w.T.dot(cov).dot(w)  # the qth centered portfolio moment
            m3 = w.T.dot(M3).dot(np.kron(w, w))
            m4 = w.T.dot(M4).dot(np.kron(np.kron(w, w), w))
            skew = m3/m2**1.5
            kurt = m4/m2**2 - 3
            z = stats.norm.ppf(alpha)  # inversed pdf to find threshold at alpha
            
            # 2nd order Cornish–Fisher expansion of the normal quantile function
            g = z + (z**2-1)/6*skew + (z**3-3*z)/24*kurt - \
                (2*z**3-5*z)/36*skew**2 
                
            def I(q):
                # q even 
                if q % 2 == 0:
                    J = lambda x: np.prod([2*j for j in np.arange(1, x+1)])
                    s = 0
                    for i in np.arange(1, q/2+1):
                        s += J(q/2) / J(i) * g**(2*i) * stats.norm.pdf(g)
                    return s + J(q/2) * stats.norm.pdf(g)
                # q odd
                else:
                    J = lambda x: np.prod([2*j+1 for j in np.arange(0, x+1)])
                    qstar = (q-1)/2
                    s = 0
                    for i in np.arange(0, qstar+1):
                        s += J(qstar) / J(i) * g**(2*i+1) * stats.norm.pdf(g)
                    return s - J(qstar) * stats.norm.cdf(g)
               
            E = -1/alpha * ( 
                stats.norm.pdf(g) + 
                (I(4) - 6*I(2) + 3*stats.norm.pdf(g))/24*kurt +    
                (I(3) - 3*I(1))/6*skew +
                (I(6) - 15*I(4) + 45*I(2) - 15*stats.norm.pdf(g))/72*skew**2
                )
            # modify E to min(E,g) in case alpha is too small and E too big 
            E[E > g] = g[E > g]
            ES = -w.T.dot(mu) - np.sqrt(m2) * E
            return ES[0, 0]

#%% Portfolio Risk: Risk Contributions
    
    def vol_c(self, weights, percentage=False, cleaning=False):
        """
        Volatility contribution of each position in a portfolio;
        """
        
        w = self._checkWeights(weights)
            
        data = self._X.copy()
        if cleaning:
            data = self.dataCleaning(method='multivariate')
        
        cov = data.cov()
        m2 = w.T.dot(cov).dot(w)
        dm2 = 2 * cov.dot(w)  # first derivative w.r.t. w
        vol_c = 0.5 * w * dm2 / np.sqrt(m2)  # risk contribution
        if percentage:
            vol_c = vol_c / vol_c.sum()
        return vol_c
        
    def var_c(self, weights, alpha=0.05, method='MCF',
              percentage=False, cleaning=False):
        """
        Value-at-Risk contribution of each component in a portfolio;
        See details in same paper as VaR;
        """
        
        assert(method in ['gaussian', 'MCF']), \
            "method should be 'gaussian' or 'MCF'!"
        
        w = self._checkWeights(weights)
            
        data = self._X.copy()
        if cleaning:
            data = self.dataCleaning(method='multivariate')

        if method == 'gaussian':
            mu = data.mean().to_frame()
            cov = data.cov()
            m2 = w.T.dot(cov).dot(w)
            dm2 = 2 * cov.dot(w)  # first derivative w.r.t. w
            z = stats.norm.ppf(alpha)
            dvar = - mu - dm2/2/np.sqrt(m2)*z
            
        else:
            mu = data.mean().to_frame()
            cov = data.cov()
            M3 = self.coskewness()
            M4 = self.cokurtosis()
            
            m2 = w.T.dot(cov).dot(w)  # the qth centered portfolio moment
            m3 = w.T.dot(M3).dot(np.kron(w, w))
            m4 = w.T.dot(M4).dot(np.kron(np.kron(w, w), w))
            
            dm2 = 2 * cov.dot(w)  # first derivative w.r.t. w
            dm3 = 3 * M3.dot(np.kron(w, w))
            dm4 = 4 * M4.dot(np.kron(np.kron(w, w), w))
            
            skew = m3/m2**1.5
            kurt = m4/m2**2 - 3
            dskew = (2*m2**1.5 * dm3 - 3*m3*m2**0.5 * dm2) / 2*m2**3
            dkurt = (m2 * dm4 - 2*m4 * dm2) / m2**3
            
            z = stats.norm.ppf(alpha)
            
            dvar_gau = - mu - dm2/2/np.sqrt(m2)*z  # gaussian var contribution
            dvar = dvar_gau + dm2/np.sqrt(m2) * (
                   -(z**2-1)/12*skew - (z**3-3*z)/48*kurt + (2*z**3-5*z)/72*skew**2
                                                ) + np.sqrt(m2) * (
                   -(z**2-1)/6*dskew - (z**3-3*z)/24*dkurt +
                   (2*z**3-5*z)/18*skew*dskew)
       
        var_c = w * dvar
        if percentage:
            var_c = var_c / var_c.sum()
            
        return var_c

    def es_c(self, weights, alpha=0.05, method='MCF',
             percentage=False, cleaning=False):
        """
        The contribution of CVaR of each component in portfolio;
        """
        
        assert(method in ['gaussian', 'MCF']), \
            "method should be 'gaussian' or 'MCF'!"
        
        w = self._checkWeights(weights)
            
        data = self._X.copy()
        if cleaning:
            data = self.dataCleaning(method='multivariate')

        if method == 'gaussian':
            mu = data.mean().to_frame()
            cov = data.cov()
            m2 = w.T.dot(cov).dot(w)
            dm2 = 2 * cov.dot(w)  # first derivative w.r.t. w
            z = stats.norm.ppf(alpha)
            des = - mu + dm2/2/np.sqrt(m2)/alpha*stats.norm.pdf(z)
            
        else:
            mu = data.mean().to_frame()
            cov = data.cov()
            M3 = self.coskewness()
            M4 = self.cokurtosis()
            
            m2 = w.T.dot(cov).dot(w)  # the qth centered portfolio moment
            m3 = w.T.dot(M3).dot(np.kron(w, w))
            m4 = w.T.dot(M4).dot(np.kron(np.kron(w, w), w))
            
            dm2 = 2 * cov.dot(w)  # first derivative w.r.t. w
            dm3 = 3 * M3.dot(np.kron(w, w))
            dm4 = 4 * M4.dot(np.kron(np.kron(w, w), w))
            
            skew = m3/m2**1.5
            kurt = m4/m2**2 - 3
            dskew = (2*m2**1.5 * dm3 - 3*m3*m2**0.5 * dm2) / 2*m2**3
            dkurt = (m2 * dm4 - 2*m4 * dm2) / m2**3
            
            z = stats.norm.ppf(alpha)
            
            # 2nd order Cornish–Fisher expansion around normal quantile function
            g = z + (z**2-1)/6*skew + (z**3-3*z)/24*kurt - \
                (2*z**3-5*z)/36*skew**2 
                
            dg = (z**2-1)/6*dskew + (z**3-3*z)/24*dkurt - \
                 (2*z**3-5*z)/18*skew*dskew
                
            def I(q):
                # q even 
                if q % 2 == 0:
                    J = lambda x: np.prod([2*j for j in np.arange(1, x+1)])
                    s = 0
                    for i in np.arange(1, q/2+1):
                        s += J(q/2) / J(i) * g**(2*i) * stats.norm.pdf(g)
                    return s + J(q/2) * stats.norm.pdf(g)
                # q odd
                else:
                    J = lambda x: np.prod([2*j+1 for j in np.arange(0, x+1)])
                    qstar = (q-1)/2
                    s = 0
                    for i in np.arange(0, qstar+1):
                        s += J(qstar) / J(i) * g**(2*i+1) * stats.norm.pdf(g)
                    return s - J(qstar) * stats.norm.cdf(g)
                
            def dI(q):
                # q even 
                if q % 2 == 0:
                    J = lambda x: np.prod([2*j for j in np.arange(1, x+1)])
                    s = 0
                    for i in np.arange(1, q/2+1):
                        s += J(q/2)/J(i) * g**(2*i-1) * \
                             (2*i-g**2) * stats.norm.pdf(g)
                    return s - J(q/2) * g * stats.norm.pdf(g)
                # q odd
                else:
                    J = lambda x: np.prod([2*j+1 for j in np.arange(0, x+1)])
                    qstar = (q-1)/2
                    s = 0
                    for i in np.arange(0, qstar+1):
                        s += J(qstar)/J(i) * g**(2*i) * \
                             (2*i+1-g**2) * stats.norm.pdf(g)
                    return s - J(qstar) * stats.norm.pdf(g)

            E = -1/alpha * ( 
                stats.norm.pdf(g) + 
                (I(4) - 6*I(2) + 3*stats.norm.pdf(g))/24*kurt +    
                (I(3) - 3*I(1))/6*skew +
                (I(6) - 15*I(4) + 45*I(2) - 15*stats.norm.pdf(g))/72*skew**2
                )
            # modify E to min(E,g) in case alpha is too small and E too big 
            E[E > g] = g[E > g]
            
            des = -mu - dm2/2/np.sqrt(m2)*E + np.sqrt(m2)/alpha * ( 
                  (I(4) - 6*I(2) + 3*stats.norm.pdf(g)) / 24 * dkurt +
                  (I(3) - 3*I(1)) / 6 * dskew +
                  (I(6) - 15*I(4) + 45*I(2) - 15*stats.norm.pdf(g))/36*skew*dskew +
                  dg * (- g*stats.norm.pdf(g) + 
                        (dI(4) - 6*dI(2) - 3*g*stats.norm.pdf(g)) / 24 * kurt +
                        (dI(3) - 3*dI(1)) / 6 * skew +
                        (dI(6)-15*dI(4)+45*dI(2)+15*g*stats.norm.pdf(g))/72*skew**2
                        )
                    )
      
        es_c = w * des
        if percentage:
            es_c = es_c / es_c.sum()
            
        return es_c

#%% Portfolio Risk: Other Risk Metrics

    def wpc(self, weights):
        """
        Weighted portfolio correlation;
        The average pairwise correlation so that portfolio variance is equal
        to original variance after substituting all pairwise correlations with 
        this constant;
        """
        
        w = self._checkWeights(weights)[:, 0]
        
        data = self._X.values
        sig = data.std(axis=0)
        cov = np.cov(data.T)
        
        n = data.shape[1]
        s1 = 0
        s2 = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s1 += w[i] * w[j] * cov[i, j]
                s2 += w[i] * w[j] * sig[i] * sig[j] 
        return s1/s2
        
    def dr(self, weights):
        """
        Diversified ratio;
        Weighted average vol / portfolio vol
        """
        
        w = self._checkWeights(weights)
        
        data = self._X.values
        sig = data.std(axis=0).reshape(-1, 1)
        cov = np.cov(data.T)
        
        psig = np.sqrt(w.T.dot(cov).dot(w))  # portfolio vol
        wav = np.sum(w*sig)  # weighted average vol
        return (wav/psig)[0, 0]

    def cr(self, weights):
        """
        Concentrated ratio;
        A simple measure of portfolio concentration that only takes into 
        account the volatility of each asset;
        A fully concentrated long-only portfolio with only one asset has unit 
        CR, while an InvVol portfolio has the lowest CR (which equals to the 
        inverse of the number of assets it contains)
        """
        
        w = self._checkWeights(weights)
        
        data = self._X.values
        sig = data.std(axis=0).reshape(-1, 1)
        
        s2wv = (w*sig).T.dot(w*sig)  # sum of squared weighted vol
        swv2 = (np.sum(w*sig))**2  # squared sum of weighted vol
        return (s2wv/swv2)[0, 0]
    
    def wptd(self, weights, marginal='empirical'):
        """
        Weighted portfolio tail dependency coefficient;
        Essentially the weighted pairwise tail dependence, without taking into 
        account asset volatility.
        """
        
        assert(marginal in ['empirical', 'gaussian', 't', 'MCF']), \
            "marginal method should be 'empirical', 'gaussian', 't' or 'MCF'!"
        
        w = self._checkWeights(weights)[:, 0]
        
        tdm = self.tdm(marginal).values
        
        n = self._X.shape[1]
        s1 = 0
        s2 = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s1 += w[i] * w[j] * tdm[i, j]
                s2 += w[i] * w[j] 
        return s1/s2
