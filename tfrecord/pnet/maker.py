import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from util import _float_feature,_int64_feature,_byte_feature
import tensorflow as tf
def make_positive_negtive():
    import os
    file_paths = []
    labels = []

    filename = '../../tfrecord/onet/face.tfrecords'
    write = tf.python_io.TFRecordWriter(filename)

    negtive = '/home/dataset/FDDB/MTCNN_USED_DATA/negtive_face'
    positive = '/home/dataset/FDDB/MTCNN_USED_DATA/positive_face'
    for f in os.listdir(negtive):
        file_paths.append(os.path.join(negtive,f))
        labels.append([1,0])
    for f in os.listdir(positive):
        file_paths.append(os.path.join(positive,f))
        labels.append([0,1])
    index = list(range(0,len(file_paths)))

    np.random.shuffle(index)
    for i in tqdm(index):
        file = file_paths[i]
        label = labels[i]
        img = cv2.cvtColor(cv2.resize(cv2.imread(file), dsize=(48, 48)), cv2.COLOR_BGR2RGB)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": _int64_feature(list(label)),
            "img": _byte_feature([img.tostring()])
        }))
        write.write(example.SerializeToString())
