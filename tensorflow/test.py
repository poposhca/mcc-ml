import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
p = tf.matmul(matrix1,matrix2)

W_layer1 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="W_layer1")

with tf.Session() as sess:
    res = sess.run(p)
    print(res)