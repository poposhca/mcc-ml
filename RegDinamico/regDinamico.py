import numpy as np   
import pandas as pd
import matplotlib.pyplot as pt
import random
from sklearn import linear_model as lin
from sklearn.cross_validation import train_test_split 

df = pd.read_csv("/Users/icloud/OneDrive/MCC/ML/proyectos/RegDinamico/regLinPoli.csv")
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)

#Normalizar
X_train = (X_train - X_train.mean()) / (X_train.max() - X_train.min())
Y_train = (Y_train - Y_train.mean()) / (Y_train.max() - Y_train.min())
#Constantes
eta = 0.1
x0 = 1
#Valor maximo del random
rmax = 10
#Pesos iniciales
ws = [random.randint(0,rmax) for i in range (0,X_train.shape[1] + 1)]
#Error por cada iteracion
err = []

overflow = False

for i in range (0,X_train.shape[0]):
    v = ws[0]*x0 + (ws[1:]*X_train.iloc[i]).sum()
    try:
        t = (Y_train.iloc[i] - v)
        err.append(t**2)
    except OverflowError:
        overflow = True
        break
    ws[0] = ws[0] + x0 * t
    ws[1:] = [ws[j] + X_train.iloc[i].iloc[j-1] * t * eta for j in range(1,len(ws))]
    print ws

#Imprimir el error
if overflow: print "Hubo error"
else: print "No hubo overflow"
pt.plot(err)
pt.show()

#pt.plot(X_train,Y_train)
#pt.plot([0,w[0]], [20, ])