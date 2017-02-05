import pandas as pd
from sklearn import svm
import matplotlib.pyplot as pt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

df = pd.read_csv('elecciones/facts.csv')
X = df[df.columns[6:]]
Y = df.dif_gop_dem
Y_cat = [1 if Y.iloc[i] >= 0 else 0 for i in range(len(Y))]

#Un ciclo para selccionar la mejor C, tendriamos que generar otro para la gamma

c_range = range(1,10)

rbf_scores_mean = []
for c in c_range:
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=c)
    scores = cross_val_score(rbf_svc, X, Y_cat, cv = 10, scoring = 'accuracy')
    rbf_scores_mean.append(scores.mean())

poly_scores_mean = []
for c in c_range:
    rbf_svc = svm.SVC(kernel='poly', gamma=0.7, C=c)
    scores = cross_val_score(rbf_svc, X, Y_cat, cv = 10, scoring = 'accuracy')
    poly_scores_mean.append(scores.mean())

#Imprimir las C's y seleccionar el accuaracy mas alto, el indice +1 correspondiente a
# ese valor es la C que se selecciona