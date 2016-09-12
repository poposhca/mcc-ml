import numpy as np   
import pandas as pd
import matplotlib.pyplot as pt
from sklearn import linear_model as lin
from sklearn.cross_validation import train_test_split 

flag = 1

if flag == 1:
    #Datos lineales
    df = pd.read_csv("/Users/icloud/OneDrive/MCC/ML/proyectos/LinReg/regLin.csv")
    X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
    pt.scatter(X_train,Y_train, color='red')
    #regresion linear
    reg = lin.LinearRegression()
    reg.fit(X_train, Y_train)
    #pt.plot(X_test, reg.predict(X_test), linewidth=3)
    #pt.show()
    #error

    print reg.coef_
    w0 = reg.intercept_
    w1 = np.linspace(-reg.coef_[0], reg.coef_[0]*3,100)
    # p = reg.intercept_+reg.coef_[0]*X_test -- Lo que esta haciendo predict, regresa el punto y de la recta
    def error (w0,w1):
        y = w0 + w1 * X_test
        e = y - Y_test
        return e
    pt.plot(w1,[np.mean(error(i,w0)) for i in w1], color = 'red')
    pt.show()


elif flag == 2:
    #Datos no lineales
    df = pd.read_csv("/Users/icloud/OneDrive/MCC/ML/proyectos/LinReg/regLin2.csv")
    X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
    pt.scatter(X_train,Y_train, color='red')
    #Se eleva al cuadrado el Y
    reg = lin.LinearRegression()
    reg.fit(X_train**2, Y_train)
    x=np.linspace(0,100,101)
    pt.plot(x, [reg.predict(i**2) for i in x], color = 'blue', linewidth = 3)
    pt.show()

elif flag == 3:
    df = pd.read_csv("/Users/icloud/OneDrive/MCC/ML/proyectos/LinReg/regLin3.csv")
    X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
    pt.scatter(X_train,Y_train, color='red')
    #Transformacion senoidal
    reg = lin.LinearRegression()
    reg.fit(np.sin(X_train),Y_train)
    x=np.linspace(0,100,101)
    pt.plot(x, [reg.predict(np.sin(i)) for i in x], color = 'blue', linewidth = 3)
    pt.show()