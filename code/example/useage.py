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

# how to use the tensorflow queue
def example3():
    q = tf.FIFOQueue(2,"int32")
    init = q.enqueue_many(([1,2],))
    x = q.dequeue()
    y = x + 1
    q_inc = q.enqueue([y])
    with tf.Session() as sess:
        sess.run(init)
        while 1:
            v,_= sess.run([x,q_inc])
            print(v)

# how to use threading

def example4():
    import numpy as np
    def run(coord,id):
        while not coord.should_stop():
            print(id)
    import threading
    c=tf.train.Coordinator()
    ts = [threading.Thread(target=run,args=(c,i)) for i in range(0,5)]
    for t in ts:t.start()
    c.join(ts)
if __name__ == '__main__':
    example4()