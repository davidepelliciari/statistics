import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
#from scipy.interpolate import interp1d
import scipy.optimize
import math

# !!! IMPORTANCE SAMPLING !!!
# Vogliamo determinare l'integrale di e^(-x^2) tra 0 e 1
#
# Metodo 1. Importance Sampling con pdf a scelta A*e^-x
# Metodo 2. Sampling da una distribuzione uniforme

# Dichiarazioni
a = 0
b = 1
e = math.e
Num = 10000   #Numero di samples da generare

truth = 0.7468241328    #valore vero dell'integrale, calcolato con wolfram Alpha

# Dichiarazione funzioni

def fun(x):     #integranda
    return e**(-x**2)



def P(x):       #pdf candidata per l'importance sampling
    A = e/(e-1)
    return A*e**(-x)



def randdist(x, pdf, nvals):    #genera un sample distribuito secondo una pdf a scelta (P(x))
    """Produce nvals random samples from pdf(x), assuming constant spacing in x."""

    # get cumulative distribution from 0 to 1
    cumpdf = np.cumsum(pdf)
    cumpdf *= 1/cumpdf[-1]

    # input random values
    randv = np.random.uniform(size=nvals)

    # find where random values would go
    idx1 = np.searchsorted(cumpdf, randv)
    # get previous value, avoiding division by zero below
    idx0 = np.where(idx1==0, 0, idx1-1)
    idx1[idx0==0] = 1

    # do linear interpolation in x
    frac1 = (randv - cumpdf[idx0]) / (cumpdf[idx1] - cumpdf[idx0])
    randdist = x[idx0]*(1-frac1) + x[idx1]*frac1

    return randdist



def integral(x,N):      #formula per lo stimatore di I secondo importance sampling
    sum = 0
    for i in range (0,N):
        sum = sum + (fun(x[i])/P(x[i]))
    return (1/N)*sum


x = np.linspace(a,b,Num)

x_unif = np.random.rand(Num)

randdist_vals = randdist(x,P(x),Num)

# PRINT DEI RISULTATI OTTENUTI E DISCOSTAMENTO DAL VALORE VERO
print("Valore calcolato (IMP SAMP): ", integral(randdist_vals,Num), "Discostamento:", (integral(randdist_vals,Num)-truth)/truth)
print("Valore calcolato (UNIFORM SAMP): ", integral(x_unif,Num), "Discostamento: ", (integral(x_unif,Num)-truth)/truth)

plt.hist(randdist_vals,50, label='samples from A*e^-x')
plt.legend()
plt.show()
plt.plot(x,fun(x),color='blue', alpha=0.3, label="integranda")
plt.plot(x,P(x),color='red',alpha=0.3, label="sampler")
plt.plot(randdist_vals,fun(randdist_vals),linestyle='', marker='.',color='blue')
plt.plot(randdist_vals,P(randdist_vals),linestyle='', marker='.',color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
