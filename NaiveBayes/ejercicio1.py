# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:33:16 2016

@author: poposhca
""" 
from __future__ import division                         #En python 2.7 para que las divisiones sean floats
import math                                             #Math como en C#
import numpy as np   
import pandas as pd
from scipy.stats import norm                          #scipy contiene la paqueteria de estadistica
from sklearn.cross_validation import train_test_split   #sklearn contiene la paqueteria de seleccion de datos

#Leer archivo
df = pd.read_csv("/Users/icloud/OneDrive/MCC/ML/spambase/spambase.data")

#Dividir datos, la funcion train_test_split divide bien bonito los datos de forma aleatoria 
#X es una de las columnas, Y va a ser siempre  si es spam o no
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)

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
#La función pdf devuelve la altura de un punto de la siatribución estandar norm.pdf(x,m,std)



#pdf_spam_pre = pd.DataFrame([np.log(norm(x_spam_mean[col],x_spam_std[col]).pdf(x_test[col])) for col in x_test.columns ]).sum(axis=0)
#pdf_nospam_pre = pd.DataFrame([np.log(norm(x_nospam_mean[col],x_nospam_std[col]).pdf(x_test[col])) for col in x_test.columns ]).sum(axis=0)

print "Fin del programa"