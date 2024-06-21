
'''
 Obsolute version using NumbaQuadpack, library that interfaces efficient C quadrature library with Numba. 
 Compilation time made this less efficient than plain Numba nojit functions with scipy quadrature.
Interfacing with the library took some figuring out, leaving this here for reference.
 '''
from NumbaQuadpack import quadpack_sig, dqags
import numpy as np
from numba import njit, jit, vectorize, cfunc
from numba.types import float64, int8, FunctionType
from scipy.special import binom
import timeit
from scipy.integrate import quad as scquad

'''Utils'''

def timer(st,n = 1000):
    st+= f'took {timeit.timeit(st, globals = globals(), number = n)}'
    print(st)

'''constants'''
# @njit
def unif(x):
    return x

# @njit 
def const(x):
    return 1

# @njit 
def gpd_CDF(x,kappa, mu, sigma):
    return 1- (1 + kappa*(x - mu)/sigma)**(-1/kappa)


# @njit 
def gpd_PDF(x,kappa, mu, sigma):
    return 1/sigma*(1 + kappa*(x - mu)/sigma)**(-1/kappa -1)


''' Factories '''
def make_dist_by_param(kappa, mu, sigma):
    # @njit
    def pdf(x):
        return gpd_PDF(x,kappa, mu, sigma)
    # @njit
    def CDF(x):
        return gpd_CDF(x,kappa, mu, sigma)
    return (pdf, CDF)



# @njit
def orderCDF(x: float64[:,:], k: np.int8, n: np.int8, F):
    result = np.zeros(x.shape)
    for j in range(n-k+1,n+1):
        result += binom(float(n), float(j))*(F(x)**j)*(1- F(x))**(n-j)
        # result += nbinom(float(n),float(j))*(F(x)**j)*(1- F(x))**(n-j)
    return result


# funcsig = FunctionType(float64(float64))
# @njit(float64(float64, int8, int8, funcsig, funcsig),
        # inline = 'always')
def orderPDF(x, k, n, f, F):
    return n*binom(float(n-1), float(k-1))*(F(x)**(n-k))*(1- F(x))**(k-1)*f(x)


def integrandFactoryPartial(k, n, f, F, transform):
    # @njit
    def temp(x):
        return transform(x)*orderPDF(x, k, n, f, F)
    return temp


def make_integrand_ptr(k, n, f, F, transform):
    integrand = integrandFactoryPartial(k, n, f, F, transform)
    @cfunc(quadpack_sig)
    def hackit(x, data_):
        return integrand(x)
    return hackit.address

# Test flow

# @jit # will prob break
def sum_highest(k,n, f, F, transform, support):
    '''
    produces expected sum of k highest transform(x) for x from dist f.
    expects njitted f, F, transform. f, F for parametric gpd can be
    obtained from make_dist_by_param
    '''
    result = 0.0
    for j in range(k+1):
        integ = make_integrand_ptr(j,n,f,F, transform)
        num, _, __ = dqags(integ, support[0], support[1])
        result += num
    return result


    
def test_unif():
    f0, F0 = make_dist_by_param(-1,0,1)
    print(sum_highest(3,3,f0,F0, unif, np.array([0.0,1.0])))
    pass



if __name__ == "__main__":
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
