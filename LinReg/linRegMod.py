import numpy as np   
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.cross_validation import train_test_split 

df = pd.read_csv("regLin1.csv")
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)
pt.scatter(X_train,Y_train, color='red')

M = len(X_train)
w1 = (M * X_train.T.dot(Y_train.T) - (X_train.sum() * Y_train.sum())) / (M * ((X_train**2).sum()) - (X_train.sum())**2)
w0 = (Y_train.sum()/M)-((w1*X_train.sum())/M)

pt.scatter(X_train,Y_train)
xmax = X_train.max()
pt.plot([0,xmax],[w0,w1*xmax+w0], color = "red", linewidth = 3)
pt.show()