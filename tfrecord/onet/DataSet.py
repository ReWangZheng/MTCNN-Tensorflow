import tensorflow as tf
import numpy as np
class TrainDateSet:
    def __init__(self,Net='Onet'):
        if Net=='onet':
            self.train_size = (48,48)
        elif Net=='rnet':
            self.train_size = (24,24)
        elif Net=='pnet':
            self.train_size = (12,12)
        self.face_d = tf.data.TFRecordDataset(['/home/regan/code/deeplearn/MTCNN-tensorflow/tfrecord/{}/face.tfrecords'.format(Net)])
        self.bound_d = tf.data.TFRecordDataset(['/home/regan/code/deeplearn/MTCNN-tensorflow/tfrecord/{}/bound.tfrecords'.format(Net)])
        self.landmark_d = tf.data.TFRecordDataset(
            ['/home/regan/code/deeplearn/MTCNN-tensorflow/tfrecord/{}/landmark.tfrecords'.format(Net)])
        self.face_d = self.face_d.map(self.parse_single_face,num_parallel_calls=3).repeat().shuffle(300).batch(300).prefetch(300)
        self.bound_d = self.bound_d.map(self.parse_single_bound,num_parallel_calls=3).repeat().shuffle(300).batch(300).prefetch(300)
        self.landmark_d = self.landmark_d.map(self.parse_single_landmark,num_parallel_calls=3).repeat().shuffle(300).batch(300).prefetch(300)
        self.face_data = None
        self.bound_data = None
        self.landmark_data = None

    def init(self,sess):
        self.face_data = self.init_dataset(sess,self.face_d)
        self.bound_data = self.init_dataset(sess, self.bound_d)
        self.landmark_data = self.init_dataset(sess, self.landmark_d)
    def get_data(self,sess):
        x_face,y_face = sess.run(self.face_data)
        x_bound,y_bound = sess.run(self.bound_data)
        x_l,y_l = sess.run(self.landmark_data)
        return merge(x_face,x_bound,x_l,y_face,y_bound,y_l)
    def init_dataset(self,sess,dataset):
        data_iter = dataset.make_initializable_iterator()
        data = data_iter.get_next()
        data_init = data_iter.make_initializer(dataset)
        sess.run(data_init)
        return data
    def parse_single_face(self,serialized_example):
        """
        解析tf.record
        :param serialized_example:
        :param opposite: 是否将图片取反
        :return:
        """
        features = tf.parse_single_example(
            serialized_example,
            features={
                'img': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([2], tf.int64)
            }
        )
        tf_image = tf.decode_raw(features['img'], tf.uint8)#获得图像原始的数据
        tf_label = tf.cast(features['label'], tf.int64)
        # PS:恢复原始图像数据,reshape的大小必须与保存之前的图像shape一致,否则出错
        # tf_image=tf.reshape(tf_image, [-1])    # 转换为行向量
        tf_image =tf.reshape(tf_image, [*self.train_size,3]) # 设置图像的维度
        return tf_image, tf_label
    def parse_single_bound(self,serialized_example):
        """
        解析tf.record
        :param serialized_example:
        :param opposite: 是否将图片取反
        :return:
        """
        features = tf.parse_single_example(
            serialized_example,
            features={
                'img': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([4], tf.float32)
            }
        )
        tf_image = tf.decode_raw(features['img'], tf.uint8)#获得图像原始的数据
        tf_label = tf.cast(features['label'], tf.float32)
        # PS:恢复原始图像数据,reshape的大小必须与保存之前的图像shape一致,否则出错
        # tf_image=tf.reshape(tf_image, [-1])    # 转换为行向量
        tf_image =tf.reshape(tf_image, [*self.train_size, 3]) # 设置图像的维度
        return tf_image, tf_label
    def parse_single_landmark(self,serialized_example):
        """
        解析tf.record
        :param serialized_example:
        :param opposite: 是否将图片取反
        :return:
        """
        features = tf.parse_single_example(
            serialized_example,
            features={
                'img': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([10], tf.float32)
            }
        )
        tf_image = tf.decode_raw(features['img'], tf.uint8)#获得图像原始的数据
        tf_label = tf.cast(features['label'], tf.float32)
        # PS:恢复原始图像数据,reshape的大小必须与保存之前的图像shape一致,否则出错
        # tf_image=tf.reshape(tf_image, [-1])    # 转换为行向量
        tf_image =tf.reshape(tf_image, [*self.train_size, 3]) # 设置图像的维度
        return tf_image, tf_label

def merge(x_c_train, x_r_train, x_l_train, y_c_train, y_r_train, y_l_train):
    X = np.concatenate([x_c_train, x_r_train, x_l_train], axis=0)

    YC = np.concatenate([y_c_train, np.zeros(shape=[len(y_r_train) + len(y_l_train), y_c_train.shape[1]])],
                        axis=0)

    YR = np.concatenate([np.zeros(shape=[len(y_c_train) + len(y_l_train), y_r_train.shape[1]]), y_r_train],
                        axis=0)

    YL = np.concatenate([np.zeros(shape=[len(y_c_train) + len(y_r_train), y_l_train.shape[1]]), y_l_train],
                        axis=0)

    label = np.concatenate(
        [np.ones(shape=[len(x_c_train), ], dtype=int),
         np.zeros(shape=[len(x_r_train), ], dtype=int),
         np.ones(shape=[len(x_l_train)], dtype=int) * 2
         ], axis=0)
    return X, YC, YR, YL, label
if __name__ == '__main__':
    sess = tf.Session()
    data = TrainDateSet(Net='pnet')
    data.init(sess)
    import matplotlib.pyplot  as plt
    import cv2
    for i in range(0,100):
        X, YC, YR, YL, label = data.get_data(sess)
        index = np.where(label==2)
        for idx in index[0]:
            plt.imshow(np.array(X[idx],dtype=int))
            lm = YL[idx]
            img = X[idx]
            h, w, c = X[idx].shape
            for i in range(0,len(lm),2):
                x1 = int(lm[i] * w)
                y1 = int(lm[i + 1] * h)
                # img = cv2.circle(img, (x1, y1), 2, color=(255, 0, 0))
            plt.imshow(img)
            plt.show()