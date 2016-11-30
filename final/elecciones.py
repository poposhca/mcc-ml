import pandas as pd
import numpy as np

datos = pd.read_csv('elecciones/county_facts.csv')
resultados = pd.read_csv('elecciones/PresidentialResults.csv')
dict = pd.read_csv('elecciones/county_facts_dictionary.csv')

condados = datos[datos.fips % 1000 != 0]
newdf = pd.DataFrame()
for i in range(10):
    condado = condados[condados.fips == resultados.fips[i]]
    newCol = pd.DataFrame({'dif':[resultados.dif[i]]})
    row = condado.join(newCol)
    newdf = newdf.append(row)
newdf.to_csv('elecciones/test.csv')