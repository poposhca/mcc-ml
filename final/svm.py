import pandas as pd
from sklearn import svm
import matplotlib.pyplot as pt
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split 

df = pd.read_csv('elecciones/facts.csv')
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[6:]],df.dif_gop_dem, train_size=0.75)

Y_train_cat = [1 if Y_train.iloc[i] >= 0 else 0 for i in range(len(Y_train))]
Y_test_cat = [1 if Y_test.iloc[i] >= 0 else 0 for i in range(len(Y_test))]
C=1.0

#La regresion lineal no esta funcionando tan bien como esperaba
#linear_svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train_cat)

rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, Y_train_cat)
Y_predict_2 = rbf_svc.predict(X_test)
acc = accuracy_score(Y_test_cat,Y_predict_2)
print acc

poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, Y_train_cat)
Y_predict_3 = poly_svc.predict(X_test)
acc = accuracy_score(Y_test_cat,Y_predict_3)
print acc