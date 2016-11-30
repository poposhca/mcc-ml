import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split 

df = pd.read_csv('elecciones/facts.csv')
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[6:]],df.dif_gop_dem, train_size=0.75)

Y_test_cat = [1 if Y_test.iloc[i] >= 0 else 0 for i in range(len(Y_test))]

#Tamanio de cada capa
input_size = X_train.shape[1]
hidden_size = 2
output_size=1

##RED NEURONAL
#Input Layer
x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])
#Hidden Layer
wh = tf.Variable(tf.random_uniform([input_size, hidden_size], -1, 1))
bh = tf.Variable(tf.zeros([hidden_size]))
h = tf.nn.sigmoid(tf.matmul(x,wh)+bh)
#Output layer
wo = tf.Variable(tf.random_uniform([hidden_size,1], -1, 1))
bo = tf.Variable(tf.zeros([1]), name="b_layer1")
vo = tf.nn.sigmoid(tf.matmul(h,wo)+bo)