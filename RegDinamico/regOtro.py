import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error



df = pd.read_csv("regLin.csv")
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]])


#dfColUno = pd.Series(1., index=['a', 'b', 'c', 'd', 'e'])

arr = [1] * len(X_train)

X_train['X0'] = arr

randomA = random.uniform(0,10)
randomB = random.uniform(0,10)

w0 = []
w1 = []

ArY_train = Y_train.values

def WiXi(a,b,c,d):
    return (a * c) + (b * d)
    
error = [0] * len(Y_train)
error[0] = (ArY_train[0]) - WiXi(X_train.iloc[0,1],X_train.iloc[0,0],randomA,randomB)

eta = 0.1

w0 = [0] * len(Y_train)
w0[0] = randomA + (eta*error[0]*(X_train.iloc[0,1]))

w1 = [0] * len(Y_train)
w1[0] = randomB + (eta*error[0]*(X_train.iloc[0,0]))



for i in range(1,len(Y_train)):
    error[i]= (ArY_train[i]) - WiXi(X_train.iloc[i,1],X_train.iloc[i,0],w0[i-1],w1[i-1])
    w0[i] = w0[i-1] + (eta*error[i]*(X_train.iloc[i,1]))
    w1[i] = w1[i-1] + (eta*error[i]*(X_train.iloc[i,0]))

# ====== A partir de aqui se estandarizaron los datos
X_trainstd = preprocessing.scale(X_train['X'])

Y_trainstd = preprocessing.scale(Y_train)


randomAstd = random.uniform(0,10)
randomBstd = random.uniform(0,10)

w0std = []
w1std = []

ArY_trainstd = Y_trainstd

errorstd = [0] * len(Y_trainstd)
errorstd[0] = (ArY_trainstd[0]) - WiXi(1,X_trainstd[i],randomA,randomB)

eta = 0.1

w0std = [0] * len(Y_trainstd)
w0std[0] = randomA + (eta*errorstd[0]*(1))

w1std = [0] * len(Y_train)
w1std[0] = randomB + (eta*errorstd[0]*(X_trainstd[i]))

for i in range(1,len(Y_trainstd)):
    errorstd[i]= (ArY_trainstd[i]) - WiXi(1,X_trainstd[i],w0std[i-1],w1std[i-1])
    w0std[i] = w0std[i-1] + (eta*errorstd[i]*(1))
    w1std[i] = w1std[i-1] + (eta*errorstd[i]*(X_trainstd[i]))

plt.scatter(errorstd,w1std)

#print "w0"
#print w0

#print "w1"
#print w1

#print "error"
#print error