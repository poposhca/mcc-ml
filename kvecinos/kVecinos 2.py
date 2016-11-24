import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

centro=np.array([1,2])
def puntosCirculo( centro, npuntos, r, etiqueta):
	for i in range(npuntos):
		xnew= np.random.uniform(-r+centro[0], r+centro[0])
		ytop=np.sqrt((r**2) - (xnew-centro[0])**2)+centro[1]
		ylow=-(np.sqrt((r**2) - (xnew-centro[0])**2))+centro[1]
		ynew=np.random.uniform(ylow, ytop)
		if i ==0:
			puntos=np.array([xnew, ynew, etiqueta])
		else:
			puntos= np.vstack( (puntos, np.array([xnew, ynew, etiqueta]) ))
	
	return( puntos )

npuntos=10

centro=	np.array( [[1,1], [-1,1], [1,-1], [-1,-1]] )
puntos1=pd.DataFrame( puntosCirculo(centro[0], npuntos, 2, 1))
puntos2=pd.DataFrame(puntosCirculo(centro[1], npuntos, 2, 2))
puntos3=pd.DataFrame(puntosCirculo(centro[2], npuntos, 2, 3))
puntos4=pd.DataFrame(puntosCirculo(centro[3], npuntos, 2, 4))


data=puntos1
data=data.append(puntos2)
data=data.append(puntos3)
data=data.append(puntos4)
data.columns=['X1', 'Y1', 'label']
data=data.set_index([range(len( data['X1']))])

punto=np.array([0,0])
print('RESOLVIENDO KVECINOS')
def calcularDistancias(punto, data):
	for i in range(len(data['X1'])):
		#print(i==0)
		x1=data['X1'][i]
		#print('XXXX: {}'.format(x1))
		y1=data['Y1'][i]
		x2=punto[0]
		y2=punto[1]
		d= np.sqrt( ( x1-x2 )**2 + ( y1-y2 )**2 )
		print('distancia calculada: {}'.format(d))
		if i ==0:
			distancias=[d]
		else:
			#print('imprimir liosta: {},    d {}'.format(distancias, d))
			distancias.append(d)
			#print('imprimir liosta22: {}'.format(distancias))
	distancias=np.array(distancias)
	return(distancias)


distancias= calcularDistancias( punto, data )

print('ORDENAR USANDO DISTANCIAS')
k=5
ordenDistancias=[x for (y,x) in sorted(zip(distancias ,data.index))]
kVecinos=ordenDistancias[:k]
diccionario={}
for i in range(k):
	print(i)
	x=data['X1'][ kVecinos[i]]
	y=data['Y1'][ kVecinos[i]]
	v=data['label'][ kVecinos[i]]
	if i ==0:
		
		puntosVecinos=np.array([x,y])
		print(puntosVecinos)
	else:
		puntosVecinos=np.vstack( ( puntosVecinos,np.array([ x,y]) ) )

	if v in diccionario.keys():
		diccionario[v]=diccionario[v]+1
	else:
		diccionario[v]=1

puntosVecinos=pd.DataFrame( puntosVecinos)

fig = plt.figure()
ax = fig.add_subplot(111) 
ax.scatter(puntos1[0], puntos1[1], color='green')
ax.scatter(puntos2[0], puntos2[1], color='blue')
ax.scatter(puntos3[0], puntos3[1], color='red')
ax.scatter(puntos4[0], puntos4[1], color='yellow')
ax.scatter(punto[0], punto[1], color='gray')
ax.scatter(puntosVecinos[0], puntosVecinos[1], color='black')