import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from util import _float_feature,_int64_feature,_byte_feature
def make(dsize=(48,48)):
    data_info_path = '/home/dataset/AFLW/MTCNN/data_info.csv'
    df = pd.read_csv(data_info_path)
    file_list = df.iloc[:, 1].to_numpy()
    land_mark = df.iloc[:, 2:].to_numpy()
    filename = '../../tfrecord/pnet/landmark.tfrecords'
    write = tf.python_io.TFRecordWriter(filename)
    for idx,(file,label) in tqdm(enumerate(zip(file_list,land_mark))):
        img = cv2.cvtColor(cv2.resize(cv2.imread(file),dsize),cv2.COLOR_BGR2RGB)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":_float_feature(list(label)),
            "img":_byte_feature([img.tostring()])
        }))
        write.write(example.SerializeToString())
def make_pos():
    pass
def make_part(dsize=(48, 48)):
    part_info = '/home/dataset/FDDB/MTCNN_USED_DATA/part_info.txt'
    file_paths = []
    bounds = []
    with open(part_info) as f:
        lines = f.readlines()
        for idx in range(0, len(lines), 2):
            file_paths.append(lines[idx].strip())
            bounds.append([float(nub) for nub in lines[idx + 1].strip().split(' ')])
    filename = '../../tfrecord/pnet/bound.tfrecords'
    write = tf.python_io.TFRecordWriter(filename)
    for (file,label) in tqdm(zip(file_paths,bounds)):
        img = cv2.cvtColor(cv2.resize(cv2.imread(file), dsize), cv2.COLOR_BGR2RGB)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":_float_feature(list(label)),
            "img":_byte_feature([img.tostring()])
        }))
        write.write(example.SerializeToString())
def make_positive_negtive(save_size=(48, 48)):
    import os
    file_paths = []
    labels = []

    filename = '../../tfrecord/pnet/face.tfrecords'
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
        img = cv2.cvtColor(cv2.resize(cv2.imread(file), save_size), cv2.COLOR_BGR2RGB)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": _int64_feature(list(label)),
            "img": _byte_feature([img.tostring()])
        }))
        write.write(example.SerializeToString())
if __name__ == '__main__':
    make_part(dsize=(12, 12))
