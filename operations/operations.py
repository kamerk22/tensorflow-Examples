import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(20)
b = tf.constant(10)

with tf.Session() as sess:
    print("a: %i" % sess.run(a), "b: %i" % sess.run(b))
    print("Addition with constants: %i" % sess.run(a + b))
    print("Minus with constants: %i" % sess.run(a - b))
    print("Multiplication with constants: %i" % sess.run(a * b))
    print("Division with constants: %i" % sess.run(a / b))
    print(" ")

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("Addition with variables: %f" % sess.run(add, feed_dict={a: 30.55, b: 10.02}))
    print("Multiplication with variables: %f" % sess.run(mul, feed_dict={a: 29.23, b: 14.14}))


matrix_a = tf.constant([[[30., 37., 50.]]])
matrix_b = tf.constant([[[40.],[20.], [8.]]])

product = tf.matmul(matrix_a, matrix_b)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)