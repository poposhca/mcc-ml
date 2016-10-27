import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X=np.asarray([[0,0],[0,1],[1,0],[1,1]])
Y=np.asarray([[0],[0],[0],[1]])
input_size=2
output_layer_size=1
#Los placeholders son para los datos de entrada y salida
x = tf.placeholder(tf.float32, [None, input_size])
y_ = tf.placeholder(tf.float32, [None, output_layer_size]) #la y real
#La variables para lo que se va a ir calculando y modificando en el camino
W_layer1 = tf.Variable(tf.random_uniform([input_size,output_layer_size], -1, 1), name="W_layer1")
b_layer1 = tf.Variable(tf.zeros([output_layer_size]), name="b_layer1")  #Es el w0

#La propagacion de la red
#Es el algoritmo de la regularizacion
y = tf.nn.sigmoid(tf.matmul(x,W_layer1)+b_layer1) #la de nuestra regresion logistica
lossfn = tf.reduce_mean(tf.reduce_sum((y_-y)**2)) #cuadratico
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(lossfn)
init = tf.initialize_all_variables()
sess = tf.Session() # tipo de sesion, puede ser interactiva

#Correr el tensorflow
sess.run(init)
for i in range(1000):
  sess.run(train_step, feed_dict={x: X, y_: Y})

#Ejemplo de como ver las variables
print(sess.run(y, feed_dict={x: X}))
print(sess.run(W_layer1))
sess.close()