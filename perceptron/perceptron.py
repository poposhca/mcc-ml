import numpy as np   
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.cross_validation import train_test_split
import random

#Datos Reglin
#df = pd.read_csv("regLin4.csv")
#X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
#X_train = (X_train - X_train.mean()) / X_train.std()

#Datos and
xs = {'x1':[0,0,1,1], 'x2':[0,1,0,1]}
X_train = pd.DataFrame(xs)
Y_train = pd.Series([0,0,0,1])

#Valores regularizacion
err = []
eta = 0.01
x0 = 1
rmax = 1
larr = np.arange(0,0.1,0.005)
ws = []

#Correr varias veces para que esta cosa aprenda bien, pero no siempre funciona
for k in range(50):
    for l in larr:
        ws = [random.randint(0,rmax) for i in range (0,X_train.shape[1] + 1)]
        for i in range (0,X_train.shape[0]):
            v = ws[0]*x0 + (ws[1:]*X_train.iloc[i]).sum()
            if(v < 0): v = 0
            else: v = 1
            t = (Y_train.iloc[i] - v)
            err.append(t)
            ws[0] = (ws[0]) + x0 * t
            ws[1:] = [(ws[j]) + (X_train.iloc[i].iloc[j-1] * t * eta) - (ws[j] * l) for j in range(1,len(ws))]

pt.plot([-(ws[0]/ws[2]),-(ws[0]+ws[1]*2)/ws[2]],[0,2])
pt.scatter(X_train.x1,X_train.x2)
pt.show()