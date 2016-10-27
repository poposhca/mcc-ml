import tensorflow as tf
import numpy as np

#Funcion XOR
X=np.asarray([[0,0],[0,1],[1,0],[1,1]])
Y=np.asarray([[0],[1],[1],[0]])

#Tamanio de cada capa
input_size = 2
hidden_size = 2
output_size=1

#Placeholder (variables de tipo tensor) / Input layer
x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

#CADA LAYER CONTIENE VARIAS NEURONAS, POR ESO EL NOMBRE

#Hidden layer:
#Tenemos dos neuronas en el la capa escondida, cada una es un perceptron

W_hidden = tf.Variable(tf.random_uniform([input_size, 2], -1, 1), name="W_layer1")      #Entradas de dos neuronas
b_hidden = tf.Variable(tf.zeros([2]), name="b_layer1")                                #La salida de cada neurona
#La coneccion usando las operaciones
v1 = tf.nn.sigmoid(tf.matmul(x,W_hidden)+b_hidden)
#v1 = tf.nn.tanh(tf.matmul(x,W_hidden)+b_hidden)

#Output Layer

W_output = tf.Variable(tf.random_uniform([hidden_size,output_size], -1, 1), name="W_layer1")
b_output = tf.Variable(tf.zeros([output_size]), name="b_layer1")
#Tenemos una salida en la capa de utput
v2 = tf.nn.sigmoid(tf.matmul(v1,W_output)+b_output)
#v2 = tf.nn.tanh(tf.matmul(v1,W_output)+b_output)
#En esta capa se hace la reduccion de error
loss = tf.reduce_mean(tf.reduce_sum((v2-y)**2))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#Correr el tensorflow
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(10000):
  sess.run(train_step, feed_dict={x: X, y: Y})
#print(sess.run(v1, feed_dict={x: X}))
print(sess.run(v2, feed_dict={x: X}))
print(sess.run(W_output))
sess.close()