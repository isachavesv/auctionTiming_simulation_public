from NumbaQuadpack import quadpack_sig, dqags
import numpy as np
from numba import njit, jit, vectorize, cfunc
from numba.types import float64, int8, FunctionType
from scipy.special import binom
import timeit
from scipy.integrate import quad as scquad
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import product, repeat, chain
from joblib import Parallel, delayed, parallel_backend
from matplotlib.ticker import FormatStrFormatter

'''Utils'''

def timer(st,n = 1000):
    st+= f'took {timeit.timeit(st, globals = globals(), number = n)}'
    print(st)

'''constants'''
@njit
def unif(x):
    return x

@njit 
def const(x):
    return 1

@njit 
def gpd_CDF(x,kappa, mu, sigma):
    return 1- (1 + kappa*(x - mu)/sigma)**(-1/kappa)


@njit 
def gpd_PDF(x,kappa, mu, sigma):
    return 1/sigma*(1 + kappa*(x - mu)/sigma)**(-1/kappa -1)

@njit
def support_maker_by_param(kappa, mu, sigma):
    assert kappa <1, 'integrable only for kappa < 1'
    if kappa >= 0: result = np.array([mu, np.inf])
    else: result = np.array([mu, mu - sigma/kappa])
    return result
    

''' Factories '''
def make_dist_by_param(kappa, mu, sigma):
    assert kappa <1, 'integrable only for kappa < 1'
    @njit
    def pdf(x):
        return gpd_PDF(x,kappa, mu, sigma)
    @njit
    def CDF(x):
        return gpd_CDF(x,kappa, mu, sigma)
    return (pdf, CDF)

def make_mr_by_param(kappa, mu, sigma):
    assert kappa < 1, 'integrable only for kappa < 1'
    @njit
    def mr(x):
        return max((1 - kappa)*x - sigma + kappa*mu,0)
    return mr



@njit
def orderCDF(x: float64[:,:], k: np.int8, n: np.int8, F):
    result = np.zeros(x.shape)
    for j in range(n-k+1,n+1):
        result += binom(float(n), float(j))*(F(x)**j)*(1- F(x))**(n-j)
        # result += nbinom(float(n),float(j))*(F(x)**j)*(1- F(x))**(n-j)
    return result


funcsig = FunctionType(float64(float64))
@njit(float64(float64, int8, int8, funcsig, funcsig),
        inline = 'always')
def orderPDF(x, k, n, f, F):
    return n*binom(float(n-1), float(k-1))*(F(x)**(n-k))*(1- F(x))**(k-1)*f(x)


def integrandFactoryPartial(k, n, f, F, transform):
    @njit
    def temp(x):
        return transform(x)*orderPDF(x, k, n, f, F)
    return temp

def make_integrand_ptr(k, n, f, F, transform):
    integrand = integrandFactoryPartial(k, n, f, F, transform)
    @cfunc(quadpack_sig)
    def hackit(x, data_):
        return integrand(x)
    return hackit.address

# @jit(float64(int8, int8, funcsig, funcsig, funcsig, float64[:])) # will prob break
def sum_highest(k,n, f, F, transform, support):
    '''
    produces expected sum of k highest transform(x) for x from dist f.
    expects njitted f, F, transform. f, F for parametric gpd can be
    obtained from make_dist_by_param
    '''
    result = 0.0
    for j in range(k+1):
        integ = integrandFactoryPartial(j,n,f,F, transform)
        num = scquad(integ, support[0], support[1])[0]
        result += num
    return result
    ###  C library
    # for j in range(k+1):
        # integ = make_integrand_ptr(j,n,f,F, transform)
        # num, _, __ = dqags(integ, support[0], support[1])
        # result += num
    # return result


    
def test_unif():
    '''helper timer function'''
    f0, F0 = make_dist_by_param(-1,0,1)
    urv_param = make_integrand_ptr(1,3, f0, F0, unif)
    urv_test = make_integrand_ptr(1,3,const,unif,unif)
    @cfunc(quadpack_sig)
    def urv_plain_c(x, data_):
        return 3*binom(float(3-1), float(1-1))*(x**(1-1))*(1- x)**(3-1)*x
    plain_c_ptr = urv_plain_c.address
    opts = [
            'dqags(plain_c_ptr,0,1)',
            'dqags(urv_param, 0,1)',
            'dqags(urv_test,0,1)',
            ]
    for o in opts:
        timer(o, n= 100000)


def sequence(k,N_min,N_max, f, F, transform, support):
    '''parallel computation of highest k out of n for n in N_min to N_max'''
    assert k < N_min
    '''TODO: parallel with joblib'''
    args = product(
            [k],
            range(N_min, N_max + 1),
            [f], 
            [F],
            [transform],
            [support]
            )
    with Pool() as p:
        output = p.starmap(sum_highest, args)
        return np.array(output)


def plot_compare(krange, nrange, setup1, setup2, paint_args = None ):
    '''compares ratio of highest order k statistics for k in krange as a function of samples n in nrange.
    any two scenarios. setup are tuples of pdf, cdf, transform'''
    if isinstance(krange, int):
        krange = [krange, krange+1]
    if paint_args:
        fig, ax = paint_args
    else:
        fig, ax = plt.subplots()
    '''WARNING: hard codes 4 k's only in picture '''
    print('WARNING: hard codes 4 ks only in picture')
    styles = ['-','--','-.',':'] 
    for k in range(*krange):
        seq1 = sequence(k, *nrange, *setup1)
        seq2 = sequence(k, *nrange, *setup2)
        ax.plot(np.arange(nrange[0], nrange[1]+1), seq2/seq1, linestyle = styles[k-1], label=f'k = {k}')
    return fig, ax






def plot_by_params_gpd_bare(krange, nrange, params, paint_args = None):
    '''plot ratio of highest order statistic vs highest mr for gpd case'''
    def inc_label(x):
        delta = np.sign(params[0]*params[1] - params[2]) 
        if delta ==1:
            return r'$\frac{\partial}{\partial v}MR(v)/v < 0$'
        elif delta ==-1:
            return r'$\frac{\partial}{\partial v}MR(v)/v > 0$'

    support = support_maker_by_param(*params)
    setup1 = [*make_dist_by_param(*params), unif, support]
    setup2 = [*make_dist_by_param(*params), make_mr_by_param(*params), support]
    fig, ax = plot_compare(krange, nrange, setup1, setup2, paint_args) # setup 2 is numerator

    title = rf'$\kappa = {params[0]}$, ' + inc_label(params)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4e'))
    ax.set_title(title)
    ax.legend(fontsize = 7)



def pair_threshold_gpd(krange, nrange, kappas, eps = 1):

    fig, ax = plt.subplots(ncols = 2, nrows = len(kappas))
    args = (krange, nrange)

    for i,k in enumerate(kappas):
        pos_params = args + ((k,1/k,1 + eps),) + ((fig, ax[i,0]),)
        neg_params = args + ((k,1/k,1 - eps),) + ((fig, ax[i,1]),)
        plot_by_params_gpd_bare(*pos_params)
        plot_by_params_gpd_bare(*neg_params)
    fig.tight_layout(h_pad = 2)
    filename = 'pics/'
    filename += f'paired_gpd_eps_{eps}'.replace('.','_') + '.png'
    fig.savefig(filename)



'''Utils, individual picture plotting'''
def save_picture_gpd(krange, nrange, params):
    import time
    start = time.time()
    fig, ax = plot_by_params_gpd(krange,nrange, params)
    filename = 'pics/'
    
    filename += 'test_comp_' + '_'.join((str(l) for l in params)) + '.png'
    fig.savefig(filename)
    print(f'Took {time.time() - start} seconds')


def thresh_plot_gpd(eps = 1e-1):
    kappas = np.linspace(0.1,0.9,10)
    pos_params = zip(kappas,1/kappas,repeat(1 + eps))
    neg_params = zip(kappas,1/kappas,repeat(1 - eps))
    params = chain(pos_params, neg_params)

    args = product(
            [[1,5]],
            [[6,30]],
            params)
    for a in args:
        save_picture_gpd(*a)


def plot_by_params_gpd(krange, nrange, params, paint_args = None):
    '''plot ratio of highest order statistic vs highest mr for gpd case'''
    support = support_maker_by_param(*params)
    setup1 = [*make_dist_by_param(*params), unif, support]
    setup2 = [*make_dist_by_param(*params), make_mr_by_param(*params), support]
    title = str(dict(zip(['kappa','mu', 'sigma'], params)))
    fig, ax = plot_compare(krange, nrange, setup1, setup2, paint_args) # setup 2 is numerator
    ax.set_title(title)
    return fig, ax


if __name__ == "__main__":
    # thresh_plot_gpd()
    # save_picture_gpd([1,5],[6,30],[0.3333,3,1.1])
    # save_picture_gpd([1,5],[6,30],[0.3333,3,0.9])
    pair_threshold_gpd([1,5],[6,30], [0.1,0.5,0.9], eps = 1e-3)

