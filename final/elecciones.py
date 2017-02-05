import pandas as pd
import numpy as np

'''
datos = pd.read_csv('elecciones/county_facts.csv')
resultados = pd.read_csv('elecciones/PresidentialResults.csv')
dict = pd.read_csv('elecciones/county_facts_dictionary.csv')
'''
datos = pd.read_csv('elecciones/facts.csv')
datos.INC110213 = datos['INC110213']/1000
datos.MAN450207 = datos['MAN450207']/100000
datos.WTN220207 = datos['WTN220207']/100000
datos.RTN131207 = datos['RTN131207']/100000

datos.to_csv('elecciones/divididos.csv')