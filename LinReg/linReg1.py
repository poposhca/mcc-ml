import numpy as np   
import pandas as pd
import matplotlib.pyplot as pt
from sklearn import linear_model as lin
from sklearn.cross_validation import train_test_split 

flag = False

if flag:
    #Datos lineales
    df = pd.read_csv("/Users/icloud/OneDrive/MCC/ML/LinReg/regLin.csv")
    X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
    pt.scatter(X_train,Y_train, color='red')
    #regresion linear
    reg = lin.LinearRegression()
    reg.fit(X_train, Y_train)
    pt.plot(X_test, reg.predict(X_test), linewidth=3)
    pt.show()

else:
    #Datos no lineales
    df = pd.read_csv("/Users/icloud/OneDrive/MCC/ML/LinReg/regLin2.csv")
    X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
    pt.scatter(X_train,Y_train, color='red')
    #Se eleva al cuadrado el Y
    reg = lin.LinearRegression()
    reg.fit(X_train**2, Y_train)
    pt.plot(X_test, reg.predict(X_test**2), linewidth=1)
    pt.show()
