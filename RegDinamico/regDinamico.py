import numpy as np   
import pandas as pd
import matplotlib.pyplot as pt
from sklearn import linear_model as lin
from sklearn.cross_validation import train_test_split 

df = pd.read_csv("/Users/icloud/OneDrive/MCC/ML/proyectos/LinReg/regLin.csv")
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[0:-1]],df[df.columns[-1]], train_size=0.75)

#Constantes
eta = 0.05
wo = 1
X_test