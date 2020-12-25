from mtcnn.layer import *
from mtcnn import config


def Pnet_factory():
    with tf.name_scope('Pnet'):
        x_input = tf.placeholder(shape=config.pnet_input_size, dtype=tf.float32)

        conv1_out = add_conv(x_input, kernel_shape=[3, 3, 3, 10], name_scope='CONV_1', activation=prelu, mode='VALID')

        pool2_out = max_pool(conv1_out, size=1, stride=2, name_scope='MP_2', mode='SAME')

        conv3_out = add_conv(pool2_out, kernel_shape=[3, 3, 10, 16], name_scope='CONV_3', activation=prelu, mode='VALID')

        conv4_out = add_conv(conv3_out, kernel_shape=[3, 3, 16, 32], name_scope='CONV_4', activation=prelu, mode='VALID')

        face_classfier = add_conv(conv4_out, kernel_shape=[1, 1, 32, 2], name_scope='CONV_5', activation=tf.nn.softmax,
                                  mode='VALID')

        bound_reg = add_conv(conv4_out, kernel_shape=[1, 1, 32, 4], name_scope='CONV_6', activation=None, mode='VALID')

        landmark_reg = add_conv(conv4_out, kernel_shape=[1, 1, 32, 10], name_scope='CONV_7', activation=None, mode='VALID')

        return x_input, tf.squeeze(face_classfier, axis=[1, 2]), tf.squeeze(bound_reg, axis=[1, 2]), tf.squeeze(
            landmark_reg, axis=[1, 2])


class Pnet:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x_input, self.face_c, self.bound_r, self.landmakr_r = Pnet_factory()
            print('ok')


class Onet:
    def __init__(self):
        pass


class Rnet:
    def __init__(self):
        pass


class MTCNN:
    def __init__(self):
        pass
if __name__ == '__main__':
    Pnet()
