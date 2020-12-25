import tensorflow as tf
import numpy as np
# this function is used to create a variable
def weight_variable(shape,name,layer_type='conv'):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(shape=shape, initializer=initial, name=name)

#this function can be used to create a bias variable
def bias_variable(shape,name=None):
    initial = tf.constant(value=0.1,shape=shape)
    return tf.Variable(initial_value=initial,name=name)

# conv2d function
def conv2d(input_v,kernel,stride,mode='SAME'):
    return tf.nn.conv2d(input_v,kernel,stride,mode)

# maxpooling function
def max_pool(input_v,size,stride,name_scope,mode="SAME"):
    with tf.variable_scope(name_scope):
        return tf.nn.max_pool(input_v,[1,size,size,1],[1,stride,stride,1],mode)

# add conv layer
def add_conv(inputs,kernel_shape,name_scope,activation=tf.nn.relu,s=1,mode='SAME'):
    w,h,c_o,c_n = kernel_shape
    with tf.variable_scope(name_scope):
        w_conv = weight_variable(shape=[w, h, c_o, c_n],
                                       name="w")
        b_conv = bias_variable(shape=[c_n], name="b")
        if activation is not None:
            out = activation(conv2d(inputs, w_conv, stride=[1, s, s, 1],mode=mode) + b_conv)
        else:
            out = conv2d(inputs, w_conv, stride=[1, s, s, 1], mode=mode) + b_conv
        return out
def add_dense(inputs,units,name_scope,activation=None):
    with tf.variable_scope(name_scope):
        shape_list = inputs.shape.as_list()
        w = weight_variable(shape=[shape_list[1], units], name="w", layer_type="fallten")
        b = bias_variable(shape=[units], name="b")
        out = tf.matmul(inputs, w) + b
        if activation is None:
            return out
        else:
            return activation(out)

# flatten layer
def add_flatten(input_tensor,name):
    shape_list = input_tensor.shape.as_list()
    n,w,h,c = shape_list
    return tf.reshape(input_tensor,shape=[-1,w * h * c],name=name)

def prelu(_x):
    # prelu.x is a global variable,set to prevent renaming
    if not hasattr(prelu, 'x'):
        prelu.x = 0
    prelu.x+=1
    # this variable can be used,when the x is negtive
    alphas = tf.get_variable('alpha{}'.format(prelu.x), _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
    # the first step is to directly caculate the value that was activated by relu
    pos = tf.nn.relu(_x)
    # if x > 0,neg will be a zero.
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos+neg


