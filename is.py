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
B = 0.5
e = math.e
Num = 10000   #Numero di valori da generare

truth = 0.7468241328    #valore vero dell'integrale, calcolato con wolfram Alpha

# ----------- Dichiarazione funzioni

def fun(x):     #integranda
    return e**(-x**2)



def P(x, B):       #pdf candidata per l'importance sampling
    if(B==0):
        A = 1
    else:
        A = B/(1-e**(-B))

    return A*e**(-B*x)


def integral(x,N,B):      #formula per lo stimatore di I secondo importance sampling
    sum = 0
    for i in range (0,N):
        sum = sum + (fun(x[i])/P(x[i],B))
    return (1/N)*sum


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

# ------------------

x = np.linspace(a,b,Num)
x_unif = np.random.rand(Num)

bins = 50   #quanti valori per B?

results = np.zeros(bins)
Disc = np.zeros(bins)
Bs = np.zeros(bins)

B = 0

for i in range(0,bins):
    B += 5/bins
    Bs[i] = B   #Array che salva i valori di B
    randdist_vals = randdist(x,P(x,B),Num)
    int = integral(randdist_vals, Num,B)
    results[i] = int
    Disc[i] = np.abs((int - truth)/truth)


# Distribuzione dei risultati (integrali) in funzione del valore di B
plt.subplot(1,2,1)
plt.title("Risultati in funzione di B")
plt.plot(Bs,results, marker='.', linestyle='',color='k')
plt.plot(Bs,truth + Bs*0, color='darkred', linestyle='--', alpha=0.5, label="Valore vero")
plt.plot(Bs, integral(x_unif, Num, 0) + Bs*0, color='seagreen',alpha=0.3, label="Dist. uniforme")
plt.xlabel("B")
plt.ylabel("Results")
plt.legend()
#plt.ylim(0.7,0.8)

plt.subplot(1,2,2)
plt.title("Errori in funzione di B")
plt.plot(Bs,Disc, marker='.', linestyle='',color='k')
plt.xlabel("B")
plt.ylabel("Error")
plt.show()

# -------------- FIND THE MINIMUM ERROR

min = results[1]
val = 0

for i in range(2,bins):
    if(Disc[i] < min):
        min = Disc[i]
        val = i
    else:
        continue

print("\n")
print("\n")
print("!!! IMPORTANCE SAMPLING !!!")
print("\n")
print("Ho svolto un'analisi su quale fosse la migliore funzione da cui prendere il sample di valori.")
print("\n")
print("La funzione e' nella forma: A e^Bx. Facendo variare B da 0.5 a 5 in un range di 50 valori si ottiene:")
print("\n")
print("Valore migliore trovato per l'integrale: ", results[val], "\n")
print("Valore vero pari a: ", truth, "\n")
print("B: ", Bs[val], "\n")
print("con errore: ", Disc[val], "\n")


B = 0
for i in range(0,bins):
    B+=5/bins
    plt.plot(x,P(x,B),color='red',alpha=0.3)

plt.plot(x,fun(x),color='k', label="integranda")
plt.plot(x,P(x,Bs[val]), color='darkred', alpha=0.7, label='sampler max')
plt.ylim(0.2,1.05)
plt.legend()
plt.show()


#DISTRIBUZIONE DEI RISULTATI a B fissato: confronto tra le due distribuzioni

N_sim = 500
ris = np.zeros(N_sim)
ris_un = np.zeros(N_sim)
sim_number = np.zeros(N_sim)

for i in range(0,N_sim):
    x_vals = randdist(x, P(x,1), Num)
    x_unif = np.random.rand(Num)
    ris[i] = integral(x_vals,Num,1)
    ris_un[i] = integral(x_unif, Num, 0)
    sim_number[i] = i

plt.title("Distribuzione e confronto dei risultati con " + str(N_sim) +" samples")
plt.hist(ris,50, label='Dist. a scelta, B=1')
plt.hist(ris_un,50, color='darkred', label='Dist. Uniforme', alpha=0.5)
plt.legend()
plt.show()

plt.title("Distribuzione e confronto dei risultati con " + str(N_sim) +" samples")
plt.plot(sim_number,ris,marker='.', linestyle='',color='k',label="Dist. a scelta, B=1")
plt.plot(sim_number,ris_un, marker='.', linestyle='', label="Dist. uniforme")
plt.legend()
plt.show()

print("\n")
print("!!! CONFRONTO RISULTATI TRA DISTRIBUZIONE A SCELTA (B=1) E DISTRIBUZIONE UNIFORME !!! ")
print("\n")
print("DISTRIBUZIONE A SCELTA (IMPORTANCE SAMPLING): ")
print("\n")
print("Valore medio: ", np.mean(ris))
print("\n")
print("Varianza: ", np.std(ris)**2)
print("\n")
print("\n")
print("DISTRIBUZIONE UNIFORME: ")
print("\n")
print("Valore medio: ", np.mean(ris_un))
print("\n")
print("Varianza: ", np.std(ris_un)**2)
