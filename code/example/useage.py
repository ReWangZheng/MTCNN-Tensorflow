#example 1:how to use tf.squess()
import tensorflow as tf
def example1():
    t = tf.constant([[1],[3],[5]])
    squeeze = tf.squeeze(t)
    print(t.shape.as_list())
    with tf.Session() as sess:
        print(sess.run(squeeze))
#example 2:how to use tf.where() and how to use tf.nn.top_k()
def example2():
    v1 = tf.constant([1,2,3,4,5])
    v2 = tf.constant([1,4,2,0,5])
    v3 = tf.constant([7,8,9,10,0])
    v4 = tf.constant([11, 12, 13, 14,8])
    res = tf.where(tf.equal(v1,v2),v3,v4)
    res2 = tf.nn.top_k(v4,3)
    with tf.Session() as sess:
        print(sess.run(res))
        print(sess.run(res2))



if __name__ == '__main__':
    example2()