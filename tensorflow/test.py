import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
p = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    res = sess.run(p)
    print(res)