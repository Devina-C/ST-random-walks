import logging
from time import time
import warnings

import numpy as np
import scipy as sp
from scipy import interpolate
from scipy import optimize
from scipy import stats
from findiff import Diff

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm

import pandas as pd
logging.basicConfig(level=logging.DEBUG)


def qvalue(pv, pi0=None):
    '''
    Estimates q-values from p-values

    This function is modified based on https://github.com/nfusi/qvalue

    Args
    ====
    pi0: if None, it's estimated as suggested in Storey and Tibshirani, 2003.

    '''
    assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

    original_shape = pv.shape
    pv = pv.ravel()  # flattens the array in place, more efficient than flatten()

    m = float(len(pv))

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pv) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = np.arange(0, 0.90, 0.01)
        counts = np.array([(pv > i).sum() for i in np.arange(0, 0.9, 0.01)])
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        # fit natural cubic spline
        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)

        if pi0 > 1:
            pi0 = 1.0

    assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    p_ordered = np.argsort(pv)
    pv = pv[p_ordered]
    qv = pi0 * m/len(pv) * pv
    qv[-1] = min(qv[-1], 1.0)

    for i in range(len(pv)-2, -1, -1):
        qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

    # reorder qvalues
    qv_temp = qv.copy()
    qv = np.zeros_like(qv)
    qv[p_ordered] = qv_temp

    # reshape qvalues
    qv = qv.reshape(original_shape)

    return qv


def get_l_limits(X):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    R_vals = np.unique(R2.flatten())
    R_vals = R_vals[R_vals > 1e-8]
    
    l_min = np.sqrt(R_vals.min()) / 2.
    l_max = np.sqrt(R_vals.max()) * 2.
    
    return l_min, l_max


def SE_kernel(X, l):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.exp(-R2 / (2 * l ** 2))


def linear_kernel(X):
    K = np.dot(X, X.T)
    return K / K.max()


def cosine_kernel(X, p):
    ''' Periodic kernel as l -> oo in [Lloyd et al 2014]

    Easier interpretable composability with SE?
    '''
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.cos(2 * np.pi * np.sqrt(R2) / p)


def gower_scaling_factor(K):
    ''' Gower normalization factor for covariance matric K

    Based on https://github.com/PMBio/limix/blob/master/limix/utils/preprocess.py
    '''
    n = K.shape[0]
    P = np.eye(n) - np.ones((n, n)) / n
    KP = K - K.mean(0)[:, np.newaxis]
    trPKP = np.sum(P * KP)

    return trPKP / (n - 1)


def factor(K):
    S, U = np.linalg.eigh(K)
    # .clip removes negative eigenvalues
    return U, np.clip(S, 1e-8, None)


def get_UT1(U):
    return U.sum(0)


def get_UTy(U, y):
    return y.dot(U)


def mu_hat(delta, UTy, UT1, S, n, Yvar=None):
    ''' ML Estimate of bias mu, function of delta.
    '''
    if Yvar is None:
        Yvar = np.ones_like(S)

    UT1_scaled = UT1 / (S + delta * Yvar)
    sum_1 = UT1_scaled.dot(UTy)
    sum_2 = UT1_scaled.dot(UT1)

    return sum_1 / sum_2


def s2_t_hat(delta, UTy, S, n, Yvar=None):
    ''' ML Estimate of structured noise, function of delta
    '''
    if Yvar is None:
        Yvar = np.ones_like(S)

    UTy_scaled = UTy / (S + delta * Yvar)
    return UTy_scaled.dot(UTy) / n


def LL(delta, UTy, UT1, S, n, Yvar=None):
    ''' Log-likelihood of GP model as a function of delta.

    The parameter delta is the ratio s2_e / s2_t, where s2_e is the
    observation noise and s2_t is the noise explained by covariance
    in time or space.
    '''

    mu_h = mu_hat(delta, UTy, UT1, S, n, Yvar)
    
    if Yvar is None:
        Yvar = np.ones_like(S)

    sum_1 = (np.square(UTy - UT1 * mu_h) / (S + delta * Yvar)).sum()
    sum_2 = np.log(S + delta * Yvar).sum()

    with np.errstate(divide='ignore'):
        return -0.5 * (n * np.log(2 * np.pi) + n * np.log(sum_1 / n) + sum_2 + n)


def make_objective(UTy, UT1, S, n, Yvar=None):
    def LL_obj(log_delta):
        return -LL(np.exp(log_delta), UTy, UT1, S, n, Yvar)

    return LL_obj


def lbfgsb_max_LL(UTy, UT1, S, n, Yvar=None):
    LL_obj = make_objective(UTy, UT1, S, n, Yvar)
    min_boundary = -10
    max_boundary = 20.
    x, f, d = optimize.fmin_l_bfgs_b(LL_obj, 0., approx_grad=True,
                                                 bounds=[(min_boundary, max_boundary)],
                                                 maxfun=64, factr=1e12, epsilon=1e-4)
    max_ll = -f
    max_delta = np.exp(x[0])

    boundary_ll = -LL_obj(max_boundary)
    if boundary_ll > max_ll:
        max_ll = boundary_ll
        max_delta = np.exp(max_boundary)

    boundary_ll = -LL_obj(min_boundary)
    if boundary_ll > max_ll:
        max_ll = boundary_ll
        max_delta = np.exp(min_boundary)


    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n, Yvar)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, S, n, Yvar)
    
    x0 = np.log(max_delta)
    h = 1e-5  # step size (same idea as dx in derivative)
    x = np.linspace(x0 - 5*h, x0 + 5*h, 11)  # 11 points around x0 for central diff
    # Evaluate the function at each grid point
    f = np.array([LL_obj(xi) for xi in x])
    d2 = Diff(0, h)  
    second_deriv = d2(f)[5] 

    s2_logdelta = 1. / (second_deriv ** 2)

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat, s2_logdelta


def make_FSV(UTy, S, n, Gower):
    def FSV(log_delta):
        s2_t = s2_t_hat(np.exp(log_delta), UTy, S, n)
        s2_t_g = s2_t * Gower

        return s2_t_g / (s2_t_g + np.exp(log_delta) * s2_t)

    return FSV


def lengthscale_fits(exp_tab, U, UT1, S, Gower, num=64):
    ''' Fit GPs after pre-processing for particular lengthscale
    '''
    results = []
    n, G = exp_tab.shape
    for g in tqdm(range(G), leave=False):
        y = exp_tab.iloc[:, g]
        UTy = get_UTy(U, y)

        t0 = time()
        max_reg_ll, max_delta, max_mu_hat, max_s2_t_hat, s2_logdelta = lbfgsb_max_LL(UTy, UT1, S, n)
        max_ll = max_reg_ll
        t = time() - t0

        # Estimate standard error of Fraction Spatial Variance
        FSV = make_FSV(UTy, S, n, Gower)
        
        x0 = np.log(max_delta)
        h = 1e-5
        x = np.linspace(x0 - 5*h, x0 + 5*h, 11)
        f = np.array([FSV(xi) for xi in x])
        d1 = Diff(0, h)
        first_deriv = d1(f)[5] 
        s2_FSV = first_deriv ** 2 * s2_logdelta
        
        results.append({
            'g': exp_tab.columns[g],
            'max_ll': max_ll,
            'max_delta': max_delta,
            'max_mu_hat': max_mu_hat,
            'max_s2_t_hat': max_s2_t_hat,
            'time': t,
            'n': n,
            'FSV': FSV(np.log(max_delta)),
            's2_FSV': s2_FSV,
            's2_logdelta': s2_logdelta
        })
        
    return pd.DataFrame(results)


def null_fits(exp_tab):
    ''' Get maximum LL for null model
    '''
    results = []
    n, G = exp_tab.shape
    for g in range(G):
        y = exp_tab.iloc[:, g]
        max_mu_hat = 0.
        max_s2_e_hat = np.square(y).sum() / n  # mll estimate
        max_ll = -0.5 * (n * np.log(2 * np.pi) + n + n * np.log(max_s2_e_hat))

        results.append({
            'g': exp_tab.columns[g],
            'max_ll': max_ll,
            'max_delta': np.inf,
            'max_mu_hat': max_mu_hat,
            'max_s2_t_hat': 0.,
            'time': 0,
            'n': n
        })
    
    return pd.DataFrame(results)


def const_fits(exp_tab):
    ''' Get maximum LL for const model
    '''
    results = []
    n, G = exp_tab.shape
    for g in range(G):
        y = exp_tab.iloc[:, g]
        max_mu_hat = y.mean()
        max_s2_e_hat = y.var()
        sum1 = np.square(y - max_mu_hat).sum()
        max_ll = -0.5 * ( n * np.log(max_s2_e_hat) + sum1 / max_s2_e_hat + n * np.log(2 * np.pi) )

        results.append({
            'g': exp_tab.columns[g],
            'max_ll': max_ll,
            'max_delta': np.inf,
            'max_mu_hat': max_mu_hat,
            'max_s2_t_hat': 0.,
            'time': 0,
            'n': n
        })
    
    return pd.DataFrame(results)



def get_mll_results(results, null_model='const'):
    null_lls = results.query('model == "{}"'.format(null_model))[['g', 'max_ll']]
    model_results = results.query('model != "{}"'.format(null_model))
    model_results = model_results[model_results.groupby(['g'])['max_ll'].transform(max) == model_results['max_ll']]
    mll_results = model_results.merge(null_lls, on='g', suffixes=('', '_null'))
    mll_results['LLR'] = mll_results['max_ll'] - mll_results['max_ll_null']

    return mll_results


def dyn_de(X, exp_tab, kernel_space=None):
    if kernel_space == None:
        kernel_space = {
            'SE': [5., 25., 50.]
        }

    results = []

    if 'null' in kernel_space:
        result = null_fits(exp_tab)
        result['l'] = np.nan
        result['M'] = 1
        result['model'] = 'null'
        results.append(result)

    if 'const' in kernel_space:
        result = const_fits(exp_tab)
        result['l'] = np.nan
        result['M'] = 2
        result['model'] = 'const'
        results.append(result)

    logging.info('Pre-calculating USU^T = K\'s ...')
    US_mats = []
    t0 = time()
    if 'linear' in kernel_space:
        K = linear_kernel(X)
        U, S = factor(K)
        gower = gower_scaling_factor(K)
        UT1 = get_UT1(U)
        US_mats.append({
            'model': 'linear',
            'M': 3,
            'l': np.nan,
            'U': U,
            'S': S,
            'UT1': UT1,
            'Gower': gower
        })

    if 'SE' in kernel_space:
        for lengthscale in kernel_space['SE']:
            K = SE_kernel(X, lengthscale)
            U, S = factor(K)
            gower = gower_scaling_factor(K)
            UT1 = get_UT1(U)
            US_mats.append({
                'model': 'SE',
                'M': 4,
                'l': lengthscale,
                'U': U,
                'S': S,
                'UT1': UT1,
                'Gower': gower
            })

    if 'PER' in kernel_space:
        for period in kernel_space['PER']:
            K = cosine_kernel(X, period)
            U, S = factor(K)
            gower = gower_scaling_factor(K)
            UT1 = get_UT1(U)
            US_mats.append({
                'model': 'PER',
                'M': 4,
                'l': period,
                'U': U,
                'S': S,
                'UT1': UT1,
                'Gower': gower
            })

    t = time() - t0
    logging.info('Done: {0:.2}s'.format(t))

    logging.info('Fitting gene models')
    n_models = len(US_mats)
    for i, cov in enumerate(tqdm(US_mats, desc='Models: ')):
        result = lengthscale_fits(exp_tab, cov['U'], cov['UT1'], cov['S'], cov['Gower'])
        result['l'] = cov['l']
        result['M'] = cov['M']
        result['model'] = cov['model']
        results.append(result)

    n_genes = exp_tab.shape[1]
    logging.info('Finished fitting {} models to {} genes'.format(n_models, n_genes))

    results = pd.concat(results, sort=True).reset_index(drop=True)
    results['BIC'] = -2 * results['max_ll'] + results['M'] * np.log(results['n'])

    return results


def run(X, exp_tab, kernel_space=None):
    ''' Perform SpatialDE test

    X : matrix of spatial coordinates times observations
    exp_tab : Expression table, assumed appropriatealy normalised.

    The grid of covariance matrices to search over for the alternative
    model can be specifiec using the kernel_space paramter.
    '''
    if kernel_space == None:
        l_min, l_max = get_l_limits(X)
        kernel_space = {
            'SE': np.logspace(np.log10(l_min), np.log10(l_max), 10),
            'const': 0
        }

    logging.info('Performing DE test')
    results = dyn_de(X, exp_tab, kernel_space)
    mll_results = get_mll_results(results)

    # Perform significance test
    mll_results['pval'] = 1 - stats.chi2.cdf(mll_results['LLR'], df=1)
    mll_results['qval'] = qvalue(mll_results['pval'])

    return mll_results