import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split 

#Este metodo, asi com esta en el codigo no funciona bien

df = pd.read_csv('elecciones/facts.csv')
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[6:]],df.dif_gop_dem, train_size=0.75)

for i in range(7):
    k = KMeans(n_clusters=i+1, random_state=0).fit(X_train)
    Y_predict = k.predict(X_test)
    print Y_predict