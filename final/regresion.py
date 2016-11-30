import pandas as pd
import matplotlib.pyplot as pt
from sklearn import linear_model as lin
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split 

df = pd.read_csv('elecciones/facts.csv')
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[6:]],df.dif_gop_dem, train_size=0.75)

reg = lin.LinearRegression()
reg.fit(X_train, Y_train)
pesos = reg.coef_
print 'Pesos del modelo:'
print pesos

Y_predict = reg.predict(X_test)
Y_predict_cat = [1 if Y_predict[i] >= 0 else 0 for i in range(len(Y_predict))]
Y_test_cat = [1 if Y_test.iloc[i] >= 0 else 0 for i in range(len(Y_test))]
acc = accuracy_score(Y_test_cat,Y_predict_cat)
print '\nAccuaracy:'
print acc

print '\nMatriz de confucion:'
print confusion_matrix(Y_test_cat,Y_predict_cat)

print '\nRecall Score:'
print recall_score(Y_test_cat,Y_predict_cat)

'''
Tenemos un problema con este modelo con los Democratas, en la matriz de 
confucion hay un importante porcentaje de falsos positivos, mas o menos es 1/3
de votos que se van a los Republicanos
'''