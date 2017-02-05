import pandas as pd
import matplotlib.pyplot as pt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

df = pd.read_csv("elecciones/facts.csv")
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[6:]],df.dif_gop_dem, train_size=0.75)

Y_train_cat = [1 if Y_train.iloc[i] >= 0 else 0 for i in range(len(Y_train))]
Y_test_cat = [1 if Y_test.iloc[i] >= 0 else 0 for i in range(len(Y_test))]

numarboles = 10
forestclf = RandomForestClassifier(n_estimators=numarboles)
forestclf = forestclf.fit(X_train, Y_train_cat)
Y_forest = forestclf.predict(X_test)

print 'Numeros arboles ' + str(numarboles)
print 'Random Forest: ' + str(accuracy_score(Y_forest, Y_test_cat))
print confusion_matrix(Y_test_cat,Y_forest)