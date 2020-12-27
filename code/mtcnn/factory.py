from mtcnn.layer import *
from mtcnn import config

def Pnet_factory():
    with tf.name_scope('pnet'):
        x_input = tf.placeholder(shape=config.pnet_input_size, dtype=tf.float32)

        conv1_out = add_conv(x_input, kernel_shape=[3, 3, 3, 10], name_scope='CONV_1', activation=prelu, mode='VALID')

        pool2_out = max_pool(conv1_out, size=1, stride=2, name_scope='MP_2', mode='SAME')

        conv3_out = add_conv(pool2_out, kernel_shape=[3, 3, 10, 16], name_scope='CONV_3', activation=prelu,
                             mode='VALID')

        conv4_out = add_conv(conv3_out, kernel_shape=[3, 3, 16, 32], name_scope='CONV_4', activation=prelu,
                             mode='VALID')

        face_classfier = add_conv(conv4_out, kernel_shape=[1, 1, 32, 2], name_scope='CONV_5', activation=tf.nn.softmax,
                                  mode='VALID')

        bound_reg = add_conv(conv4_out, kernel_shape=[1, 1, 32, 4], name_scope='CONV_6', activation=None, mode='VALID')

        landmark_reg = add_conv(conv4_out, kernel_shape=[1, 1, 32, 10], name_scope='CONV_7', activation=None,
                                mode='VALID')

        return x_input, tf.squeeze(face_classfier, axis=[1, 2]), tf.squeeze(bound_reg, axis=[1, 2]), tf.squeeze(
            landmark_reg, axis=[1, 2])

def Rnet_factory():
    with tf.name_scope('Rnet'):
        x_input = tf.placeholder(shape=config.onet_input_size, dtype=tf.float32)
        conv1_out = add_conv(x_input, kernel_shape=[3, 3, 3, 28], name_scope='CONV_1', activation=prelu, mode='VALID')

        pool2_out = max_pool(conv1_out, size=1, stride=2, name_scope='MP_2', mode='SAME')

        conv3_out = add_conv(pool2_out, kernel_shape=[3, 3, 28, 48], name_scope='CONV_3', activation=prelu,
                             mode='VALID')

        pool3_out = max_pool(conv3_out, size=1, stride=2, name_scope='MP_3', mode='SAME')


        conv4_out = add_conv(pool3_out, kernel_shape=[3, 3, 48, 64], name_scope='CONV_4', activation=prelu,
                             mode='VALID')

        conv4_flatten = add_flatten(conv4_out, 'fallten')

        feature = add_dense(inputs=conv4_flatten, units=128, activation=prelu, name_scope='FC_5')

        face_classfier = add_dense(inputs=feature, units=2, activation=tf.nn.softmax, name_scope='OUT_CLASSIFER')

        bound_reg = add_dense(inputs=feature, units=4, name_scope='OUT_REG')

        landmark_reg = add_dense(inputs=feature, units=10, name_scope='OUT_LANDMARK')

        return x_input, face_classfier, bound_reg, landmark_reg

def Onet_factory():
    with tf.name_scope('Onet'):
        x_input = tf.placeholder(shape=config.rnet_input_size, dtype=tf.float32)

        conv1_out = add_conv(x_input, kernel_shape=[3, 3, 3, 32], name_scope='CONV_1', activation=prelu, mode='VALID')

        pool2_out = max_pool(conv1_out, size=1, stride=2, name_scope='MP_2', mode='SAME')

        conv3_out = add_conv(pool2_out, kernel_shape=[3, 3, 32, 64], name_scope='CONV_3', activation=prelu,
                             mode='VALID')
        pool3_out = max_pool(conv3_out, size=1, stride=2, name_scope='MP_3', mode='SAME')

        conv4_out = add_conv(pool3_out, kernel_shape=[3, 3, 64, 64], name_scope='CONV_4', activation=prelu,
                             mode='VALID')
        pool4_out = max_pool(conv4_out, size=1, stride=2, name_scope='MP_4', mode='SAME')

        conv5_out = add_conv(pool4_out, kernel_shape=[3, 3, 64, 128], name_scope='CONV_5', activation=prelu,
                             mode='VALID')

        conv5_flatten = add_flatten(conv5_out, 'fallten')

        feature = add_dense(inputs=conv5_flatten, units=256, activation=prelu, name_scope='FC_5')

        face_classfier = add_dense(inputs=feature, units=2, name_scope='OUT_CLASSIFER')

        bound_reg = add_dense(inputs=feature, units=4, name_scope='OUT_REG')

        landmark_reg = add_dense(inputs=feature, units=10, name_scope='OUT_LANDMARK')
        return x_input, face_classfier, bound_reg, landmark_reg

def Classifier_loss_factory(label, target, feature):
    mul_loss = tf.nn.softmax_cross_entropy_with_logits(logits=feature, labels=target)
    valid_index = tf.cast(tf.equal(label, 1), tf.int64)
    valid_number = tf.cast(valid_index, tf.float32)
    loss = tf.reduce_sum(tf.multiply(mul_loss, valid_number)) / tf.reduce_sum(valid_number)
    prob = tf.multiply(tf.squeeze(tf.slice(tf.nn.softmax(feature), [0, 1], [-1, 1])), valid_number)
    sel = tf.where(tf.equal(valid_index, 1))
    acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.gather(tf.argmax(target, axis=1), sel), tf.gather(tf.argmax(feature, axis=1), sel)),
                dtype=tf.float32))
    return acc, prob, loss

def Bounding_box_loss_factory(label, feature, y_bound):
    mul_loss = tf.reduce_sum(tf.square(feature - y_bound), axis=1)
    valid_index = tf.equal(label, 0)
    valid_number = tf.cast(valid_index, tf.float32)
    loss = tf.reduce_sum(tf.multiply(mul_loss, valid_number)) / tf.reduce_sum(valid_number)
    return loss

def Landmark_loss_factory(label, feature, y_landmark):
    mul_loss = tf.reduce_sum(tf.square(feature - y_landmark), axis=1)
    valid_index = tf.equal(label, 2)
    valid_number = tf.cast(valid_index, tf.float32)
    loss = tf.reduce_sum(tf.multiply(mul_loss, valid_number)) / tf.reduce_sum(valid_number)
    return loss
