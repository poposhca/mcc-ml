# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:33:16 2016

@author: poposhca
""" 
from __future__ import division                         #En python 2.7 para que las divisiones sean floats
import math                                             #Math como en C#
import numpy as np   
import pandas as pd
from scipy.stats import norm                            #scipy contiene la paqueteria de estadistica
from sklearn.cross_validation import train_test_split   #sklearn contiene la paqueteria de seleccion de datos
from sklearn.naive_bayes import GaussianNB as nb        #Algoritmo de Naive Bayes

#Leer archivo
df = pd.read_csv("/Users/icloud/OneDrive/MCC/ML/spambase/spambase.data")

#Dividir datos, la funcion train_test_split divide bien bonito los datos de forma aleatoria 
#X es una de las columnas, Y va a ser siempre  si es spam o no
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:57]],df[df.columns[-1]], train_size=0.75)

#Calculo del prior
#NOTA: Lo que hace pandas es equivalente a: X_train[i] for i in range(len(X_train)) if Y_train[i] == 1
spamset = Y_train == 1
nspamset = Y_train == 0

#Dado spam
pspam = len(Y_train[spamset]) / len(Y_train)
logpspam = math.log(pspam)

#Dado no spamse
pnspam = len(nspamset) / len(Y_train)
logpnspam = math.log(pnspam)

#Calcular la media y la divición estandar para calcular la probabilidad de clase y la probabilidad total
#Se asume que son distribuciones normales
spammean = X_train[spamset].mean()
spamstd = X_train[spamset].std()

nspammean = X_train[nspamset].mean()
nspamstd = X_train[nspamset].std()

#Ya entrenamos el modelo, ahora hay que evaluarlo con el set de entrenaminto
#La función pdf devuelve la altura de un punto de la distribucion estandar norm.pdf(x,m,std)
a = pd.DataFrame([np.log(norm.cdf(X_test[i],loc = spammean[i], scale = spamstd[i])) for i in X_test.columns]).sum()
b = pd.DataFrame([np.log(norm.cdf(X_test[i],loc = nspammean[i], scale = nspamstd[i])) for i in X_test.columns]).sum()
spam = a > b

#Checar los resultados del algoritmo contra el algoritmo en sklearn
model = nb()
model.fit(X_train,Y_train)
res = model.predict(X_test)

check = []
for i in range(len(spam)):
    if (spam[i] == True and res[i] == 1) or (spam[i] == False and res[i] == 0): check.append(i)
print 'Programa vs SKlearn: ' + str(len(check)/len(spam))

#Checar si de verdad los modelos le arinaron usando el sklearn
from sklearn.metrics import accuracy_score
print 'Modelo programado: ' + str(accuracy_score(Y_test,spam))
print 'Modelo sklearn: ' + str(accuracy_score(Y_test,res))