import pandas as pd

dict = pd.read_csv('Data/msDataDict.csv')
data = pd.read_csv('Data/msData.csv')

print data.columns
print dict.columns

usuarios = data[data.Type == 'C']
print 'Numero de usuarios :' + str(len(usuarios))
numdata = len(data) - len(usuarios)
print 'Numero de datos: ' + str(numdata)

#Transformacion de los datos
#Se quita el valor de tipo de dato(.Type) y se reemplaza el ultimo (.Num)
#por el numero de usuario

user = data.Num[0]
pages = []
users = []
for i in range(1,len(data)):
    row = data.iloc[i]
    if(row.Type == 'C'):
        user = row.Num
    elif(row.Type == 'V'):
        pages.append(row.Page)
        users.append(user)
newdf = pd.DataFrame({'page':pages,'user':users})
newdf.to_csv('Data/newformat.csv')
