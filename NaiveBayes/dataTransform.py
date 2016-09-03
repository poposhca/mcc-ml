import pandas as pd

df = pd.read_csv('/Users/icloud/OneDrive/MCC/ML/spambase/spambase.data')
print df['0.32']
#T traspone la matriz y asi puede eliminar los repetidos sin problema
df = df.T.drop_duplicates().T
print df['0.32']
df.to_csv('/Users/icloud/OneDrive/MCC/ML/spambase/spambase.data')
