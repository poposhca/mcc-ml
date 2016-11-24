import pandas as pd
from sklearn import svm
import matplotlib.pyplot as pt
from sklearn.cross_validation import train_test_split

df = pd.read_csv("andsvm.csv")
X = df[['X1','X2']]
y = df['y']

clf = svm.SVC(kernel='linear')
clf.fit(X, y)
pt.scatter(X['X1'],X['X2'])
ws = clf.coef_[0]
print ws
pt.plot([0,2],[(ws[0]),(ws[0]+ws[1]*2)])
pt.show()