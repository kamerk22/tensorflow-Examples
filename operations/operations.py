import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant(2)
b - tf.constant(5)

with tf.Session() as sess:
    print "a: %i" % sess.run(a), "b: %i" % sess.run(b)
    print "Addition with constants: %i" % sess.run(a+b)
    print "Multiplication with constants: %i" % sess.run(a*b)
    print "Division with constants: %i" % sess.run(a/b)