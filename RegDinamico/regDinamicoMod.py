from sklearn.cross_validation import train_test_split 
import matplotlib.pyplot as pt
import pandas as pd
import numpy as np   
import random

#ESTE CODIGO SOLO FUNCIONA CON REGLIN

df = pd.read_csv("regLin.csv")
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
X_train = (X_train - X_train.mean()) / X_test.std()
Y_train = (Y_train - Y_train.mean()) / Y_train.std()

eta = 0.1
x0 = 1
rmax = 1
ws = [random.randint(0,rmax) for i in range (0,X_train.shape[1] + 1)]
for i in range (0,X_train.shape[0]):
    v = ws[0]*x0 + (ws[1:]*X_train.iloc[i]).sum()
    t = (Y_train.iloc[i] - v)
    ws[0] = ws[0] + x0 * t
    ws[1:] = [ws[j] + X_train.iloc[i].iloc[j-1] * t * eta for j in range(1,len(ws))]
    xmin = (X_train).min()
    xmax = (X_train).max()
    pt.plot([xmin,xmax],[ws[0]+ws[1]*xmin,ws[0]+ws[1]*xmax], linewidth = 0.5)
pt.scatter(X_train,Y_train)
pt.plot([xmin,xmax],[ws[0]+ws[1]*xmin,ws[0]+ws[1]*xmax],color = 'red', linewidth = 3)
pt.show()