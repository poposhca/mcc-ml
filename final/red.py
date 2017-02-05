import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split 

df = pd.read_csv('elecciones/facts.csv')
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[6:]],df.dif_gop_dem, train_size=0.75)

Y_test_cat = pd.Series([[1 if Y_test.iloc[i] >= 0 else 0] for i in range(len(Y_test))])
print Y_test