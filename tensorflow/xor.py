import tensorflow as tf
import numpy as np

X=np.asarray([[0,0],[0,1],[1,0],[1,1]])
Y=np.asarray([[0],[1],[1],[0]])
input_size=2
output_layer_size=1

x = tf.placeholder(tf.float32, [None, input_size])
h = tf.placeholder(tf.float32, [None, ])
y_ = tf.placeholder(tf.float32, [None, output_layer_size])