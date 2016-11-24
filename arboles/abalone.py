import os
import pandas as pd
import matplotlib.pyplot as pt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

df = pd.read_csv("abalone.csv")

X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[1:]],df[df.columns[0]], train_size=0.75)

#Arbol normal
treeclf = tree.DecisionTreeClassifier()
treeclf = treeclf.fit(X_train, Y_train)
Y_tree = treeclf.predict(X_test)
#Esto es para visualizar
'''
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(treeforestclf, out_file=f)
os.system("dot -Tpdf iris.dot -o iris.pdf")
'''
#Random RandomForest
forestclf = RandomForestClassifier(n_estimators=50)
forestclf = forestclf.fit(X_train, Y_train)
Y_forest = forestclf.predict(X_test)
#Accuaracy
print 'Arbol normal: ' + str(accuracy_score(Y_tree, Y_test))
print 'Random Forest: ' + str(accuracy_score(Y_forest, Y_test))