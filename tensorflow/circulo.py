import random
import numpy as np
import tensorflow as tf

#Funciones

def circUni(x, y):
    r = (x**2 + y**2)**0.5
    return r<=1

def generarDatos(num):
    xrange = [-2,2]
    #yrange = [-2,2]
    ps = np.asarray([[random.uniform(xrange[0], xrange[1]) for i in range(2)] for j in range(num)])
    zs = []
    for j in range(num):
        r = 1.0 if circUni(ps[j,0],ps[j,1]) else 0.0
        zs.append([r])
    zs = np.asarray(zs)
    return [ps, zs]

#Red neuronal
train = generarDatos(200)
test = generarDatos(10)
capas = 4

#Placeholder / Input layer
p = tf.placeholder(tf.float32, [None, 2])
z = tf.placeholder(tf.float32, [None, 1])
#Hidden layer
wh = tf.Variable(tf.random_uniform([2, capas], -1, 1))
bh = tf.Variable(tf.zeros([capas]))
h = tf.nn.sigmoid(tf.matmul(p,wh)+bh)
#Output layer
wo = tf.Variable(tf.random_uniform([capas,1], -1, 1))
bo = tf.Variable(tf.zeros([1]), name="b_layer1")
vo = tf.nn.sigmoid(tf.matmul(h,wo)+bo)
#Error y definicion de entrenamientos
loss = tf.reduce_mean(tf.reduce_sum((vo-z)**2))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#corre la sesion
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
#entrenamiento
for i in range(1000):
    sess.run(train_step, feed_dict={p:train[0], z:train[1]})
print 'Datos de pruebas:'
print test[0]
#pruebas
print 'Resultados de la red:'
print(sess.run(vo, feed_dict={p:test[0]}))
sess.close()