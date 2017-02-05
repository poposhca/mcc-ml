import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as pt
from sklearn.cross_validation import train_test_split

df = pd.read_csv("andsvm.csv")
X = df[['X1','X2']]
y = df['y']

#Entrenamiento
Cvar = 10
clf = svm.SVC(kernel='linear', C=Cvar)
clf.fit(X, y)

#Graficas
ws = clf.coef_
w =[clf.intercept_[0], ws[0][0], ws[0][1]]
m = w[1]/w[2]
V = clf.support_vectors_
pt.scatter(X.X1, X.X2)
x = np.linspace(0.0,2.0,5)
for i in range (0,len(V)):
    pt.plot(x, - m *(x-V[i][0])+V[i][1], linewidth=i)
pt.plot(x,-x*w[1]/w[2]-w[0]/w[2])
pt.show()