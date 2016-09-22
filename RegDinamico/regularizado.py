import numpy as np   
import pandas as pd
import matplotlib.pyplot as pt
import random
from sklearn import linear_model as lin
from sklearn.cross_validation import train_test_split 

#Error cuadratico medio
def errorccMedio(ws, xs, ys):
    res = 0
    for i in range (0,xs.shape[0]):
        y = ws[0] + (ws[1:] * xs.iloc[i]).sum()
        res += y - ys.iloc[i]
    return (res/xs.shape[0])**2

df = pd.read_csv("/Users/icloud/OneDrive/MCC/ML/proyectos/RegDinamico/regLinPoli.csv")
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)

X_train = (X_train - X_train.mean()) / (X_train.max() - X_train.min())
Y_train = (Y_train - Y_train.mean()) / (Y_train.max() - Y_train.min())
eta = 0.01
x0 = 1
rmax = 10

#Nueva variable l, ahora vamos a iterar para distintos valores de l
larr = np.arange(0,0.1,0.007)
err = []

for l in larr:
    ws = [random.randint(0,rmax) for i in range (0,X_train.shape[1] + 1)]
    for i in range (0,X_train.shape[0]):
        v = ws[0]*x0 + (ws[1:]*X_train.iloc[i]).sum()
        t = (Y_train.iloc[i] - v)
        ws[0] = (ws[0]) + x0 * t
        ws[1:] = [(ws[j]) + (X_train.iloc[i].iloc[j-1] * t * eta) - (ws[j] * l) for j in range(1,len(ws))]
    err.append(errorccMedio(ws,X_train,Y_train))


pt.plot(err)
pt.show()