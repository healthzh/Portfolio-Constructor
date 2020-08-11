# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:35:50 2019

@author: Victor.Long
"""

#%%
# Import Libs

import pandas as pd
import scipy.optimize as sco
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

from Risk_Metrics import RiskMetrics as rm

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


#%%
# Portfolio Constructor Class

class PortfolioConstructor(object):
    """
    Portfolio construction tool;

    All inputs & outputs are in decimal format (i.e. 2% should be 0.02);

    breakdowns should be DataFrame with strategies being columns and
    assets being index;

    benchmark should be DataFrame with only one column;

    All boundaries and constraints should be list or tuple with same order as
    assets or strategies;

    When plotting, labels will be limited to first 10 characters;

    Internal computation using np array data type;
    """

    def __init__(self,
                 X,
                 breakdowns=None,
                 benchmark=None,
                 XType='return',
                 bType='return',
                 frequency=12,
                 riskfree=0.02):

        # check inputs
        assert (isinstance(X, pd.DataFrame)), \
            'Please use DataFrame for X!'

        assert (XType in ['return', 'price']), \
            "XType should be 'return' or 'price'!"

        if breakdowns is not None:
            assert (isinstance(X, pd.DataFrame)), \
                'Please use DataFrame for breakdowns!'
            assert (X.shape[1] == breakdowns.shape[0]), \
                'X and breakdowns should have same number of assets!'

        if benchmark is not None:
            assert (isinstance(X, pd.DataFrame)), \
                'Please use DataFrame for benchmark!'
            assert (X.shape[0] == benchmark.shape[0]), \
                'X and benchmark should have same number of periods!'

        if XType == 'price':
            X = (X / X.shift(1) - 1).drop(X.index[0])

        if bType == 'price':
            benchmark = (benchmark / benchmark.shift(1) - 1).drop(benchmark.index[0])

        # initial inputs
        self._X = X
        self._bench = benchmark
        self._breaks = breakdowns
        self._rf = riskfree
        self._rm = rm(X, XType='return', frequency=frequency)  # RiskMetric object
        self._f = frequency

        # updated after construction
        self._weights = None
        self._returns = None
        self._cumuret = None
        self._drawdowns = None
        self._metrics = {}
        self._converged = False
        self._type = 'Not Constructed'
        self._value = None  # objective function value

    def __repr__(self):
        txt = "\n" + \
              "-" * 50 + "\n" + \
              f"Portfolio: {self._type}\n" + \
              f"Number of Assets: {self._X.shape[1]}\n" + \
              f"Number of Periods: {self._X.shape[0]}\n"

        if self._breaks is not None:
            txt += f"Number of Breakdowns: {self._breaks.shape[1]}\n"
        else:
            txt += "Number of Breakdowns: Not Provided\n"

        if self._bench is not None:
            txt += f"Benchmark: {self._bench.columns.values[0]}\n"
        else:
            txt += "Benchmark: Not Provided\n"

        txt += "-" * 50 + "\n"
        return txt

    #%%
    # Functions to call portfolio attributes

    def Weights(self):
        return self._weights

    def Drawdowns(self):
        return self._drawdowns

    def Returns(self):
        return self._returns

    def CumuReturns(self):
        return self._cumuret

    def Metrics(self):
        return self._metrics

    def IsConverged(self):
        return self._converged

    def Type(self):
        return self._type

    def Value(self):
        return self._value

    def PBreakdown(self):
        if self._breaks is None:
            print('\n')
            print('-' * 50)
            print('Strategy Matrix is not provided!')
            print('-' * 50)
            return
        else:
            b = np.dot(self._weights.T, self._breaks)
            b = pd.DataFrame(b.T,
                             index=self._breaks.columns,
                             columns=['Portfolio Breakdown'])
            return b

    #%%
    # Private functions

    def _cumu(self, perf):
        """
        calculate cumulative return
        """
        return np.cumprod(perf + 1) - 1

    def _dd(self, cumup):
        """
        calculate drawdowns
        """
        assert (cumup is not None), 'No cumulative perfs!'
        drawdown = []
        for i in range(len(cumup)):
            hwm = max(max(cumup[:i + 1]), 0)  # high water mark
            d = (cumup[i] - hwm) / (1 + hwm)  # current drawdown
            drawdown.append(d)
        return np.array(drawdown)

    def _calcB(self):
        """
        calculate benchmark cumulative return and dd
        """
        assert (self._bench is not None), 'No benchmark provided!'

        # benchmark cumu return
        cumu = self._cumu(self._bench.iloc[:, 0])
        b = cumu.to_frame()

        # benchmark drawdown
        d = pd.DataFrame(self._dd(cumu),
                         columns=self._bench.columns,
                         index=self._bench.index)
        return b, d

    def _winrate(self, perf):
        """
        calculate % of positive periods
        """
        return np.sum(perf > 0, axis=0) / perf.shape[0]

    def _initW(self, initialize):
        """
        initialize weights
        """
        assert (initialize in ['equal', 'random']), \
            "initialize should be 'equal' or 'random'!"

        noa = self._X.shape[1]

        if initialize == 'equal':
            weights = np.ones(noa) / noa
        else:
            weights = np.random.random(noa)
            weights /= np.sum(weights)
        return weights

    def _addWCons(self, wConstraints):
        """
        add weights constraints
        """
        noa = self._X.shape[1]

        assert (isinstance(wConstraints, list) or
                isinstance(wConstraints, tuple)), \
            'Unsupported data type, please use list or tuple!'

        nob = len(wConstraints)

        assert (nob == 2 or nob == noa), \
            'Boundaries should have number of 1 or the same as assets!'

        if nob == 2:
            bnds = [wConstraints for x in range(noa)]
        else:
            bnds = wConstraints

        return bnds

    def _Nlower(self, w, *args):
        """
        net exposure lower bound
        """
        lower = args[0]
        return np.sum(w) - lower

    def _Nupper(self, w, *args):
        """
        net exposure upper bound
        """
        upper = args[0]
        return upper - np.sum(w)

    def _Glower(self, w, *args):
        """
        gross exposure lower bound
        """
        lower = args[0]
        return np.sum(np.abs(w)) - lower

    def _Gupper(self, w, *args):
        """
        gross exposure upper bound
        """
        upper = args[0]
        return upper - np.sum(np.abs(w))

    def _Slower(self, w, *args):
        """
        strategy exposure lower bound
        """
        b = args[0]
        lower = args[1]
        return np.dot(w.T, b) - lower

    def _Supper(self, w, *args):
        """
        strategy exposure upper bound
        """
        b = args[0]
        upper = args[1]
        return upper - np.dot(w.T, b)

    def _Vtarget(self, w, *args):
        """
        volatility target function
        """
        cov_ann = args[0]
        targetVol = args[1]
        return w.T.dot(cov_ann).dot(w) - targetVol ** 2

    def _addNCons(self, net):
        """
        add net exposure constraints
        """
        assert (isinstance(net, list) or
                isinstance(net, tuple)), \
            'Unsupported data type, please use list or tuple!'

        assert (len(net) == 2), \
            'net constraints should be simple tuple or list with 2 elements!'

        const = [
            {'type': 'ineq',
             'fun': self._Nlower,
             'args': (net[0], None)
             },
            {'type': 'ineq',
             'fun': self._Nupper,
             'args': (net[1], None)
             }
        ]
        return const

    def _addVCons(self, const, cov_ann, targetVol):
        """
        add volatility target
        """
        c = {'type': 'eq',
             'fun': self._Vtarget,
             'args': (cov_ann, targetVol)
             }
        const.append(c)
        return const

    def _addGcons(self, const, gross):
        """
        add gross exposure constraints
        """
        assert (isinstance(gross, list) or
                isinstance(gross, tuple)), \
            'Unsupported data type, please use list or tuple!'

        assert (len(gross) == 2), \
            'gross constraints should be simple tuple or list with 2 elements!'

        const.append(
            {'type': 'ineq',
             'fun': self._Glower,
             'args': (gross[0], None)
             }
        )

        const.append(
            {'type': 'ineq',
             'fun': self._Gupper,
             'args': (gross[1], None)
             }
        )
        return const

    def _addBcons(self, const, bConstraints):
        """
        add breakdown constraints
        """
        assert (self._breaks is not None), \
            'Breakdown matrix is not provided!'

        assert (isinstance(bConstraints, list) or
                isinstance(bConstraints, tuple)), \
            'Unsupported data type, please use list or tuple!'

        noc = len(bConstraints)
        nos = self._breaks.shape[1]
        assert (noc == 2 or noc == nos), \
            'Boundaries should have number of 1 or the same as strategies!'

        if noc == 2:
            scons = [bConstraints for x in range(nos)]
        else:
            scons = bConstraints

        bm = self._breaks.values

        for i in range(nos):
            upper = scons[i][1]
            lower = scons[i][0]
            b = bm[:, i]

            const.append(
                {'type': 'ineq',
                 'fun': self._Slower,
                 'args': (b, lower)
                 }
            )

            const.append(
                {'type': 'ineq',
                 'fun': self._Supper,
                 'args': (b, upper)
                 }
            )
        return const

    def _addResults(self, weights, cov_ann, mu_ann, converged):
        """
        update portfolio results
        """
        idx = self._X.index
        col = self._X.columns
        X = self._X.values

        vol = np.sqrt(weights.T.dot(cov_ann).dot(weights))[0, 0]
        ret = weights.T.dot(mu_ann)[0, 0]
        sr = (ret - self._rf) / vol  # sharpe ratio

        perf = X.dot(weights)
        cumup = self._cumu(perf)
        dd = self._dd(cumup)
        wr = self._winrate(perf)[0]  # winning rate

        # update simple perf, cumu perf, weights, drawdowns, metrics
        self._returns = pd.DataFrame(perf, index=idx,
                                     columns=['Simple Returns'])

        self._cumuret = pd.DataFrame(cumup, index=idx,
                                     columns=['Cumulative Returns'])

        self._weights = pd.DataFrame(weights, index=col,
                                     columns=['Weights'])

        self._drawdowns = pd.DataFrame(dd, index=idx,
                                       columns=['Drawdowns'])

        self._metrics = {'Annualized Return': ret,
                         'Annualized Volatility': vol,
                         'Sharpe Ratio': sr,
                         'Max Drawdown': min(dd),
                         'Winning Rate': wr}

        self._converged = converged

    # %% User-Defined Portfolio

    def Customized(self, weights):
        '''
        customized portfolio
        '''

        assert (isinstance(weights, list) or
                isinstance(weights, tuple)), \
            'Unsupported data type, please use list or tuple!'

        assert (len(weights) == self._X.shape[1]), \
            'Weights should have same length as assets!'

        # annualize return and variance
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values

        # results
        weights = weights.reshape(-1, 1)
        converged = True
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Customized'

    # %% Equal-Weighted Portfolio

    def EqualWeighted(self, net=(1, 1)):
        '''
        simple equal weighted portfolio;
        portfolio exposure uses upbound of net
        '''

        assert (isinstance(net, list) or
                isinstance(net, tuple)), \
            'Unsupported data type, please use list or tuple!'

        assert (len(net) == 2), \
            'net constraints should be simple tuple or list with 2 elements!'

        # annualize return and variance
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values

        # number of assets
        noa = self._X.shape[1]

        # results
        weights = np.ones([noa, 1]) / noa * net[1]
        converged = True
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Equal Weighted'

    #%%
    # Mean-Variance Portfolio

    def MeanVariance(self,
                     targetVol,
                     wConstraints=(0, 1),
                     bConstraints=None,
                     initialize='equal',
                     net=(1, 1),
                     gross=None):
        '''
        mean-variance portfolio;

        can add weight constraints and strategy constraints;

        all boundaries and constraints should be list or tuple with same order
        as assets or startegies;
        '''

        # objective function to max mean
        def _mv(w, *M):
            return -w.T.dot(M)

        # annualize return and variance
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values

        # initialize weights
        weights = self._initW(initialize)

        # add boundaries for weights
        bnds = None
        if wConstraints:
            bnds = self._addWCons(wConstraints)

        # add net exposure constraints
        const = self._addNCons(net)

        # add target vol
        const = self._addVCons(const, cov_ann, targetVol)

        # add constraints for gross exposure
        if gross:
            const = self._addGcons(const, gross)

        # add constraints for strategies
        if bConstraints:
            const = self._addBcons(const, bConstraints)

        # run optimization
        result = sco.minimize(
            fun=_mv,
            x0=weights,
            args=(mu_ann),
            method='SLSQP',
            constraints=const,
            bounds=bnds
        )

        # update results
        weights = result.x.reshape(-1, 1)
        converged = result.success
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Mean-Variance ({:.2%})'.format(targetVol)
        self._value = result.fun

        # warning if not converged
        if not self._converged:
            print('\n')
            print('-' * 50)
            print('The optimization is not converged! \n')
            print('Looser constraints are recommanded!')
            print('-' * 50)

    #%%
    # Minimum Variance Portfolio

    def MinVariance(self,
                    wConstraints=(0, 1),
                    bConstraints=None,
                    initialize='equal',
                    net=(1, 1),
                    gross=None):
        """
        global minimum variance portfolio;

        can add weight constraints and strategy constraints;

        all boundaries and constraints should be list or tuple with same order
        as assets or strategies;
        """

        # objective function to minimize variance
        def _minvar(w, *V):
            return w.T.dot(V).dot(w)

        # annualized return and variance
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values

        # initialize weights
        weights = self._initW(initialize)

        # add boundaries for weights
        bnds = None
        if wConstraints:
            bnds = self._addWCons(wConstraints)

        # add net exposure constraints
        const = self._addNCons(net)

        # add constraints for gross exposure
        if gross:
            const = self._addGcons(const, gross)

        # add constraints for strategies
        if bConstraints:
            const = self._addBcons(const, bConstraints)

        # run optimization
        result = sco.minimize(fun=_minvar,
                              x0=weights,
                              args=(cov_ann),
                              method='SLSQP',
                              constraints=const,
                              bounds=bnds)

        # min vol
        minv_vol = np.sqrt(result.fun)

        # find best return for min vol
        mv = PortfolioConstructor(self._X, self._breaks)
        mv.MeanVariance(
            minv_vol,
            wConstraints,
            bConstraints,
            initialize,
            net,
            gross
        )

        # results
        weights = mv._weights.values
        converged = mv._converged
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Minimum-Variance'
        self._value = result.fun

        # warning if not converged
        if not self._converged:
            print('\n')
            print('-' * 50)
            print('The optimization is not converged! \n')
            print('Looser constraints are recommended!')
            print('-' * 50)

    #%%
    # Maximum Sharpe Portfolio

    def MaxSharpe(self,
                  wConstraints=(0, 1),
                  bConstraints=None,
                  initialize='equal',
                  net=(1, 1),
                  gross=None):
        """
        the tangent portfolio;

        can add weight constraints and strategy constraints;

        all boundaries and constraints should be list or tuple with same order
        as assets or strategies;
        """

        # objective function to max sharpe ratio
        def _ms(w, *args):
            mu = args[0]
            cov = args[1]
            rf = args[2]
            return - (w.T.dot(mu) - rf) / (np.sqrt(w.T.dot(cov).dot(w)))

        # annualize return and variance
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values

        # initialize weights
        weights = self._initW(initialize)

        # add boundaries for weights
        bnds = None
        if wConstraints:
            bnds = self._addWCons(wConstraints)

        # add net exposure constraints
        const = self._addNCons(net)

        # add constraints for gross exposure
        if gross:
            const = self._addGcons(const, gross)

        # add constraints for strategies
        if bConstraints:
            const = self._addBcons(const, bConstraints)

        # run optimization
        result = sco.minimize(
            fun=_ms,
            x0=weights,
            args=(mu_ann, cov_ann, self._rf),
            method='SLSQP',
            constraints=const,
            bounds=bnds
        )

        # update results
        weights = result.x.reshape(-1, 1)
        converged = result.success
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Maximum Sharpe'
        self._value = result.fun

        # warning if not converged
        if not self._converged:
            print('\n')
            print('-' * 50)
            print('The optimization is not converged! \n')
            print('Looser constraints are recommanded!')
            print('-' * 50)

    #%%
    # Inverse Volatility Portfolio

    def InverseVol(self, net=(1, 1)):
        """
        Inverse volatility portfolio;
        net exposure uses upper bound;
        """

        # annualize return and variance
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values
        std_ann = self._rm.volatility().values.reshape(-1, 1)

        # inversed vol list
        std_inverse = 1 / std_ann

        # results
        weights = std_inverse / sum(std_inverse) * net[1]
        converged = True
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Inverse Volatility'

    #%%
    # Risk Parity Portfolio

    def RiskParity(self,
                   wConstraints=(0, 1),
                   bConstraints=None,
                   initialize='equal',
                   net=(1, 1),
                   gross=None):
        """
        risk parity portfolio;

        equal volatility contribution;

        can add weight constraints and strategy constraints;

        all boundaries and constraints should be list or tuple with same order
        as assets or startegies;
        """

        # objective function to minimize differences in risk contribution
        def _risk_parity(w, *args):
            X = args[0]
            p = X.dot(w).flatten()  # portfolio returns
            n = X.shape[1]
            s = 0
            for i in range(n):
                for j in range(n):
                    covi = np.cov(p, X[:, i])[0, 1]
                    covj = np.cov(p, X[:, j])[0, 1]
                    s += (w[i] * covi - w[j] * covj) ** 2
            return s * 10 ** 8  # scale up output to avoid local minimum

        # annualize return and variance
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values

        # initialize weights
        weights = self._initW(initialize)

        # add boundaries for weights
        bnds = None
        if wConstraints:
            bnds = self._addWCons(wConstraints)

        # add net exposure constraints
        const = self._addNCons(net)

        # add constraints for gross exposure
        if gross:
            const = self._addGcons(const, gross)

        # add constraints for strategies
        if bConstraints:
            const = self._addBcons(const, bConstraints)

        # run optimization
        result = sco.minimize(fun=_risk_parity,
                              x0=weights,
                              args=(self._X.values, None),
                              method='SLSQP',
                              constraints=const,
                              bounds=bnds)

        # update results
        weights = result.x.reshape(-1, 1)
        converged = result.success
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Risk Parity'
        self._value = result.fun

        # warning if not converged
        if not self._converged:
            print('\n')
            print('-' * 50)
            print('The optimization is not converged! \n')
            print('Looser constraints are recommanded!')
            print('-' * 50)

    # %% Maximum Diversification Portfolio

    def MaxDivers(self,
                  wConstraints=(0, 1),
                  bConstraints=None,
                  initialize='equal',
                  net=(1, 1),
                  gross=None):
        '''
        max diversification portfolio;
        can add weight constraints and strategy constraints;
        '''

        # objective function to max diversification
        def _max_divers(w, *args):
            std = args[0]
            cov = args[1]
            a = w.T.dot(std)
            b = np.sqrt(w.T.dot(cov).dot(w))
            return -a / b

        # annualize return and variance
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values
        std_ann = self._rm.volatility().values.reshape(-1, 1)

        # initialize weights
        weights = self._initW(initialize)

        # add boundaries for weights
        bnds = None
        if wConstraints:
            bnds = self._addWCons(wConstraints)

        # add net exposure constraints
        const = self._addNCons(net)

        # add constraints for gross exposure
        if gross:
            const = self._addGcons(const, gross)

        # add constraints for strategies
        if bConstraints:
            const = self._addBcons(const, bConstraints)

        # run optimization
        result = sco.minimize(fun=_max_divers,
                              x0=weights,
                              args=(std_ann, cov_ann),
                              method='SLSQP',
                              constraints=const,
                              bounds=bnds)

        # results
        weights = result.x.reshape(-1, 1)
        converged = result.success
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Maximum Diversification'
        self._value = result.fun

        # warning if not converged
        if not self._converged:
            print('\n')
            print('-' * 50)
            print('The optimization is not converged! \n')
            print('Looser constraints are recommanded!')
            print('-' * 50)

    # %% Minimum Tail Dependency Portfolio

    def MinTD(self,
              wConstraints=(0, 1),
              bConstraints=None,
              initialize='equal',
              net=(1, 1),
              gross=None,
              copula='empirical',
              marginal='empirical',
              method='slope'):
        """
        minimum tail dependency portfolio
        can add weight constraints and strategy constraints;
        bivariate Gumbel copula;
        similar to MaxDivers portfolio;
        """

        # objective function to min tail dependency
        def _min_td(w, *args):
            std = args[0]
            stdtd = args[1]
            a = w.T.dot(std)
            b = np.sqrt(w.T.dot(stdtd).dot(w))
            return -a / b

        # annualize return and variance
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values
        std_ann = self._rm.volatility().values.reshape(-1, 1)
        std_diag = np.diag(self._rm.volatility().values)

        # tail dependency matrix and vol-sacled matrix
        td = self._rm.tdm(copula, marginal, None, method).values
        stdtd = std_diag.dot(td).dot(std_diag)

        # initialize weights
        weights = self._initW(initialize)

        # add boundaries for weights
        bnds = None
        if wConstraints:
            bnds = self._addWCons(wConstraints)

        # add net exposure constraints
        const = self._addNCons(net)

        # add constraints for gross exposure
        if gross:
            const = self._addGcons(const, gross)

        # add constraints for strategies
        if bConstraints:
            const = self._addBcons(const, bConstraints)
        global result
        # run optimization
        result = sco.minimize(fun=_min_td,
                              x0=weights,
                              args=(std_ann, stdtd),
                              method='SLSQP',
                              constraints=const,
                              bounds=bnds)

        # results
        weights = result.x.reshape(-1, 1)
        converged = result.success
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Minimum Tail Dependency'
        self._value = result.fun

        # warning if not converged
        if not self._converged:
            print('\n')
            print('-' * 50)
            print('The optimization is not converged! \n')
            print('Looser constraints are recommanded!')
            print('-' * 50)

    #%%
    # Minimum CVaR Portfolio

    def MinCVaR(self,
                wConstraints=(0, 1),
                bConstraints=None,
                initialize='equal',
                net=(1, 1),
                gross=None,
                alpha=0.05,
                method='MCF'):
        """
        min conditional VaR (ES) portfolio;
        see 'Optimization of Conditional Value-at-Risk'
        """

        # objective function to minimize CVaR
        def _min_cvar(w, *args):
            X = args[0]
            alpha = args[1]
            method = args[2]
            s = X.shape[0]
            loss = -X.dot(w)  # loss
            P = pd.DataFrame(-loss.reshape(-1, 1))  # portfolio return
            VaR = rm(P, frequency=self._f).var(alpha, method).values
            # VaR = self._rm.VaR_P(w, alpha, method)
            z = loss - VaR
            z[z < 0] = 0
            return VaR + 1 / (1 - alpha) / s * np.sum(z)

        # annualize return and variance
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values

        # initialize weights
        weights = self._initW(initialize)

        # add boundaries for weights
        bnds = None
        if wConstraints:
            bnds = self._addWCons(wConstraints)

        # add net exposure constraints
        const = self._addNCons(net)

        # add constraints for gross exposure
        if gross:
            const = self._addGcons(const, gross)

        # add constraints for strategies
        if bConstraints:
            const = self._addBcons(const, bConstraints)

        # run optimization
        result = sco.minimize(fun=_min_cvar,
                              x0=weights,
                              args=(self._X.values, alpha, method),
                              method='SLSQP',
                              constraints=const,
                              bounds=bnds)

        # update results
        weights = result.x.reshape(-1, 1)
        converged = result.success
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Minimum CVaR'
        self._value = result.fun

        # warning if not converged
        if not self._converged:
            print('\n')
            print('-' * 50)
            print('The optimization is not converged! \n')
            print('Looser constraints are recommanded!')
            print('-' * 50)

    #%%
    # Robust Minimum CVaR Portfolio

    def RMinCVaR(self,
                 wConstraints=(0, 1),
                 bConstraints=None,
                 initialize='equal',
                 net=(1, 1),
                 gross=None,
                 alpha=0.05,
                 method='MCF',
                 simulation=50,
                 aggregate='average'):
        """
        robust min conditional VaR (ES) portfolio;
        similar to min cvar, but fit and sampling from a multivariate skew-t
        distribution;
        aggregating methods:
            average(average weights over all simulations);
            conservative(select weights resulting in lowest CVaR);
            optimistic(select weights resulting in highest CVaR);
        require rpy2 in python and sn package in R
        """

        assert (aggregate in ['average', 'conservative', 'optimistic']), \
            "aggregate method should be 'average','conservative' or 'optimistic'!"

        n = self._X.shape[1]
        m = self._X.shape[0]

        weights = np.empty([n, simulation])  # container for weights
        cvar = []  # container for cvar
        converges = []  # container for if converged

        # import R package
        from rpy2 import robjects
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()

        mst = robjects.r('sn:: mst.mple')  # multivariate skew t
        rmst = robjects.r('sn:: rmst')  # random samples from mst
        vector = robjects.r('base:: as.vector')  # vector in R

        # to solve unexpected crash of python
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        # fit mst
        data = self._X.copy().values * 100  # scale up by 100 to make mle solvable
        fit = mst(y=data, opt_method="BFGS")

        # simulation
        for i in range(simulation):
            sample = np.array(
                rmst(n=m,
                     xi=vector(fit[1][0]),
                     Omega=fit[1][1],
                     alpha=fit[1][2],
                     nu=fit[1][3])
            ) / 100
            sample = pd.DataFrame(sample)

            # construct MinCVaR portfolio
            p = PortfolioConstructor(sample, self._breaks, frequency=self._f)

            p.MinCVaR(wConstraints=wConstraints,
                      bConstraints=bConstraints,
                      initialize=initialize,
                      net=net,
                      gross=gross,
                      alpha=alpha,
                      method=method)

            weights[:, i] = p.Weights().values.flatten()
            cvar.append(p.Value())
            converges.append(p.IsConverged())

        # restore setting
        numpy2ri.deactivate()
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'False'

        # aggregating
        if aggregate == 'average':
            weights = weights.mean(axis=1).reshape(-1, 1)
        elif aggregate == 'conservative':
            col = np.argmax(cvar)  # column number with worst CVaR
            weights = weights[:, col].reshape(-1, 1)
        else:
            col = np.argmin(cvar)  # column number with worst CVaR
            weights = weights[:, col].reshape(-1, 1)

        # update results
        mu_ann = self._rm.returns().values.reshape(-1, 1)
        cov_ann = self._rm.covariance().values
        converged = np.all(converges)
        self._addResults(weights, cov_ann, mu_ann, converged)
        self._type = 'Robust Minimum CVaR'

        # warning if not converged
        if not self._converged:
            print('\n')
            print('-' * 50)
            print('The optimization is not converged! \n')
            print('Looser constraints are recommanded!')
            print('-' * 50)

    #%%
    # Efficient Frontier

    def EfficientFrontier(self,
                          plot=True,
                          output=False,
                          step=0.002,
                          n=50,
                          name=True,
                          W=None,
                          initialize='equal',
                          benchmark=True):
        """
        efficient frontier;
        can add user's portfolio weights;
        assuming long-only without leverage;
        W is self-defined weights;
        """

        riskfree = self._rf

        # global minimum vol
        gmv = PortfolioConstructor(self._X, self._breaks)
        gmv.MinVariance(initialize=initialize)
        gmv_vol = gmv.Metrics()['Annualized Volatility']
        gmv_return = gmv.Metrics()['Annualized Return']

        # initialize frontier
        returns = [gmv_return]  # list of return
        vols = [gmv_vol]  # list of volatility
        sr = [(gmv_return - riskfree) / gmv_vol]  # list of sharpe ratio

        # construct frontier
        for i in range(n):
            vol = gmv_vol + i * step
            vols.append(vol)

            mv = PortfolioConstructor(self._X, self._breaks)
            mv.MeanVariance(vol, initialize=initialize)

            return_ = mv.Metrics()['Annualized Return']
            sharpe = (return_ - riskfree) / vol
            returns.append(return_)
            sr.append(sharpe)

        # tangent portfolio
        max_sharpe = max(sr)
        i = sr.index(max_sharpe)
        tangency = [returns[i], vols[i]]

        # plot
        if plot:
            # mean-std list of all underlying
            asset_vol = []
            asset_return = []
            for i in self._X.columns:
                asset_vol.append(self._X[i].std() * np.sqrt(12))
                asset_return.append(self._X[i].mean() * 12)

            # initialize figure
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax.tick_params(labelsize=8)
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Return')
            ax.set_title('Efficient Frontier')
            plt.grid(linestyle='--', linewidth=0.5)

            # plot frontier
            plt.plot(vols, returns, color='xkcd:blue', linewidth=1.5)

            # plot tangency line
            xx = [0, vol]
            yy = [riskfree, riskfree + max_sharpe * vol]
            plt.plot(xx, yy, color='xkcd:brown',
                     linestyle='dashed', linewidth=1)

            # plot gmv portfolio and tangency portfolio
            plt.scatter(gmv_vol, gmv_return, marker='.',
                        color='xkcd:red', linewidth=2)
            plt.scatter(tangency[1], tangency[0], marker='*',
                        color='xkcd:red', linewidth=2)

            # plot asset mean-variance pairs
            plt.scatter(asset_vol, asset_return, marker='x',
                        color='xkcd:pink', linewidth=0.6)
            if name:
                names = self._X.columns
                for i, txt in enumerate(names):
                    ax.annotate(txt, (asset_vol[i], asset_return[i]),
                                horizontalalignment='left',
                                verticalalignment='top',
                                fontsize=5)

            # plot current portfolio
            if W:
                cov_ann = self._X.cov() * 12
                V = np.sqrt(np.dot(W.T, np.dot(cov_ann, W)))
                R = np.mean(np.dot(self._X, W)) * 12
                plt.scatter(V, R, marker='^',
                            color='xkcd:teal', linewidth=2)

            # plot benchmark
            if benchmark and self._bench is not None:
                bV = self._bench.std().iloc[0] * np.sqrt(12)
                bR = self._bench.mean().iloc[0] * 12
                bn = self._bench.columns.values[0]
                plt.scatter(bV, bR, marker='s',
                            color='xkcd:orange', linewidth=2)

            # add legends
            text = ['Efficient Frontier',
                    'Tangency Line',
                    'Min Variance Portfolio',
                    'Tangency Portfolio',
                    'Underlyings']
            if W:
                text.append('Current Portfolio')
            if benchmark and self._bench is not None:
                text.append(bn)
            ax.legend(text, fontsize=6)

            plt.show()

        # output
        if output:
            frontier = pd.DataFrame({'Return': returns,
                                     'Volatility': vols,
                                     'Sharpe ratio': sr})
            return frontier, tangency

    #%%
    # Portfolio Summary

    def Summary(self, output=False, plot=True, benchmark=True):
        """
        summary stats
        visualize portfolio
        """

        if self._weights is None:
            print('\n')
            print('-' * 50)
            print('Portfolio is not constructed!')
            print('-' * 50)
            return

        # plot summary
        if plot:
            fig = plt.figure()
            years = mdates.YearLocator()  # every year
            months = mdates.MonthLocator()  # every month
            years_fmt = mdates.DateFormatter('%Y')

            # cumulative return
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.xaxis.set_major_locator(years)
            ax1.xaxis.set_major_formatter(years_fmt)
            ax1.xaxis.set_minor_locator(months)
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax1.tick_params(labelsize=8)
            ax1.set_title('Cumulative Return', fontsize=8)
            plt.grid(linestyle='--', linewidth=0.5)
            plt.plot(self._cumuret, color='xkcd:teal')
            text = ['Portfolio']
            if self._bench is not None and benchmark:
                b, d = self._calcB()
                bn = b.columns.values[0]  # benchmark name
                plt.plot(b, color='xkcd:grey', linewidth=1)
                text.append(bn)
            ax1.legend(text, fontsize=8)

            # weights
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.tick_params(labelsize=8, axis='x')
            ax2.tick_params(labelsize=6, axis='y')
            # ax2.invert_yaxis()  # labels read top-to-bottom
            ax2.xaxis.set_major_locator(ticker.AutoLocator())
            ax2.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax2.set_title('Weights', fontsize=8)
            ax2.xaxis.grid(linestyle='--', linewidth=0.5)
            y = np.arange(self._weights.shape[0])
            weights = self._weights['Weights']
            plt.barh(y, weights, color='xkcd:periwinkle')
            plt.yticks(y, [t[:10] for t in self._weights.index])  # 10 characters
            for i, w in enumerate(weights):
                ax2.annotate('{:.1%}'.format(w),
                             (weights[i] + 0.001, y[i]),
                             fontsize=6)

                # max drawdown
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.xaxis.set_major_locator(years)
            ax3.xaxis.set_major_formatter(years_fmt)
            ax3.xaxis.set_minor_locator(months)
            ax3.yaxis.set_major_locator(ticker.AutoLocator())
            ax3.yaxis.set_minor_locator(ticker.AutoLocator())
            ax3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax3.tick_params(labelsize=8)
            ax3.set_title('Drawdowns', fontsize=8)
            plt.grid(linestyle='--', linewidth=0.5)
            plt.plot(self._drawdowns, color='xkcd:dark pink')
            text = ['Portfolio']
            if self._bench is not None and benchmark:
                plt.plot(d, color='xkcd:grey', linewidth=1)
                text.append(bn)
            ax3.legend(text, fontsize=8)

            # strategy weights
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.tick_params(labelsize=8, axis='x')
            ax4.tick_params(labelsize=6, axis='y')
            # ax4.invert_yaxis()  # labels read top-to-bottom
            ax4.xaxis.set_major_locator(ticker.AutoLocator())
            ax4.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax4.set_title('Portfolio Breakdown', fontsize=8)
            ax4.xaxis.grid(linestyle='--', linewidth=0.5)
            y = np.arange(self._breaks.shape[1])
            breakdown = self.PBreakdown()['Portfolio Breakdown']
            plt.barh(y, breakdown, color='xkcd:peach')
            plt.yticks(y, [t[:10] for t in self._breaks.columns])
            for i, w in enumerate(breakdown):
                ax4.annotate('{:.1%}'.format(w),
                             (breakdown[i] + 0.001, y[i]),
                             fontsize=6)

            plt.tight_layout()
            plt.show()

        # output
        if output:
            summary = {'Type': self._type,
                       'Weights': self._weights,
                       'Metrics': self._metrics,
                       'Simple Returns': self._returns,
                       'Cumulative Returns': self._cumuret,
                       'Drawdowms': self._drawdowns,
                       'Is Converged': self._converged}
            return summary

    #%%
    # Advanced Risk Analytics

    def ARA(self, plot=True, output=False, alpha=0.05, method='MCF',
            percentage=True, marginal='empirical'):
        """
        Advanced Risk Analysis
        """

        if self._weights is None:
            print('\n')
            print('-' * 50)
            print('Portfolio is not constructed!')
            print('-' * 50)
            return

        # calculate risk metrics
        w = self._weights.values
        volc = self._rm.vol_c(w, percentage)  # Vol contribution
        varc = self._rm.var_c(w, alpha, method, percentage)  # VaR contribution
        esc = self._rm.es_c(w, alpha, method, percentage)  # ES contribution

        # portfolio VaR & portfolio ES
        var = rm(self.Returns(), frequency=self._f).var(alpha, method).values[0]
        es = rm(self.Returns(), frequency=self._f).es(alpha, method).values[0]

        wpc = self._rm.wpc(w)  # weighted portfolio correlation
        dr = self._rm.dr(w)  # diversification ratio
        cr = self._rm.cr(w)  # concentration ratio
        wptd = self._rm.wptd(w, marginal)  # weighted portfolio tail dependency

        if plot:
            fig = plt.figure()

            # Vol contribution
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.tick_params(labelsize=8, axis='x')
            ax1.tick_params(labelsize=6, axis='y')
            # ax1.invert_yaxis()  # labels read top-to-bottom
            ax1.xaxis.set_major_locator(ticker.AutoLocator())
            ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax1.set_title('Volatility Contribution', fontsize=8)
            ax1.xaxis.grid(linestyle='--', linewidth=0.5)
            y = np.arange(volc.shape[0])
            plt.barh(y, volc[0], color='xkcd:teal')
            plt.yticks(y, [t[:10] for t in volc.index])
            for i, w in enumerate(volc[0].values):
                ax1.annotate('{:.1%}'.format(w),
                             (volc[0].values[i] + 0.001, y[i] + 0.2),
                             fontsize=6)

            # VaR contribution
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.tick_params(labelsize=8, axis='x')
            ax2.tick_params(labelsize=6, axis='y')
            # ax2.invert_yaxis()  # labels read top-to-bottom
            ax2.xaxis.set_major_locator(ticker.AutoLocator())
            ax2.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax2.set_title('{:.0%} VaR Contribution'.format(1 - alpha), fontsize=8)
            ax2.xaxis.grid(linestyle='--', linewidth=0.5)
            y = np.arange(varc.shape[0])
            plt.barh(y, varc[0], color='xkcd:dull blue')
            plt.yticks(y, [t[:10] for t in varc.index])
            for i, w in enumerate(varc[0].values):
                ax2.annotate('{:.1%}'.format(w),
                             (varc[0].values[i] + 0.001, y[i] + 0.2),
                             fontsize=6)

            # CVaR contribution
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.tick_params(labelsize=8, axis='x')
            ax3.tick_params(labelsize=6, axis='y')
            # ax3.invert_yaxis()  # labels read top-to-bottom
            ax3.xaxis.set_major_locator(ticker.AutoLocator())
            ax3.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax3.set_title('{:.0%} ES Contribution'.format(1 - alpha), fontsize=8)
            ax3.xaxis.grid(linestyle='--', linewidth=0.5)
            y = np.arange(varc.shape[0])
            plt.barh(y, esc[0], color='xkcd:rose')
            plt.yticks(y, [t[:10] for t in esc.index])
            for i, w in enumerate(esc[0].values):
                ax3.annotate('{:.1%}'.format(w),
                             (esc[0].values[i] + 0.001, y[i] + 0.2),
                             fontsize=6)

            # portfolio metrics
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.tick_params(labelsize=8, axis='x')
            ax4.tick_params(labelsize=6, axis='y')
            # ax4.invert_yaxis()  # labels read top-to-bottom
            ax4.xaxis.set_major_locator(ticker.AutoLocator())
            ax4.set_title('Portfolio Risk Metrics', fontsize=8)
            ax4.xaxis.grid(linestyle='--', linewidth=0.5)
            metrics = [var * 100, es * 100, wpc, dr, cr, wptd]
            y = np.arange(len(metrics))
            plt.barh(y, metrics, color='xkcd:mango')
            plt.yticks(y, [t[:10] for t in
                           ['VaR*100', 'ES*100', 'WPC', 'DR', 'CR', 'WPTD']])
            for i, w in enumerate(metrics):
                ax4.annotate('{:.2f}'.format(w),
                             (metrics[i] + 0.001, y[i] + 0.2),
                             fontsize=6)

            plt.tight_layout()
            plt.show()

        # output
        if output:
            summary = {'Vol Contribution': volc,
                       'VaR Contribution': varc,
                       'ES Contribution': esc,
                       'Value at Risk': var,
                       'Expected Shortfall': es,
                       'Weighted Portfolio Correlation': wpc,
                       'Diversification Ratio': dr,
                       'Concentration Ratio': cr,
                       'Weighted Portfolio Tail Dependency': wptd}
            return summary


#%%
# Test

if __name__ == '__main__':
    from File_Reader import FileReader as fr

    names_equity = ['GS', 'JPM', 'HSBC', 'UBS',
                    'WMT', 'KO', 'PG', 'PM',
                    'NEE', 'DUK', 'D', 'SO',
                    'BA', 'CAT', 'GE', 'MMM',
                    'XOM', 'CVX', 'COP', 'SLB']

    names_sector = 'Sectors'

    names_bench = '^GSPC'

    path = '/Users/victor/Desktop/Efforts/Portfolio Constructor/Data/Equities'

    data = fr(path, names_equity).Read(
        concate=True, useCols=[0, 5], fill='pad', drop=True)

    breakdowns = fr(path, names_sector).Read(concate=False)

    benchmark = fr(path, names_bench).Read(
        concate=False, useCols=[0, 5])

    #%%

    a = PortfolioConstructor(data, breakdowns, benchmark,
                             XType='price', bType='price', frequency=252)

    a.MaxSharpe((0.01, 0.2), (0.01, 0.4))

    summary = a.Summary(True)

    a.ARA()

