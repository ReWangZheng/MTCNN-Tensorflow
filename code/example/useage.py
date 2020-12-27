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

#how to use TFRecord?
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def example5():
    import pandas as pd
    import cv2
    import numpy as np
    import tqdm
    import matplotlib.pyplot as plt
    data_info_path = '/home/dataset/AFLW/MTCNN/data_info.csv'
    df = pd.read_csv(data_info_path)
    file_list = df.iloc[:, 1].to_numpy()
    land_mark = df.iloc[:, 2:].to_numpy()
    filename = '../../tfrecord/record.tfrecords'
    write = tf.python_io.TFRecordWriter(filename)
    for idx,(file,label) in enumerate(zip(file_list,land_mark)):
        print(idx)
        img = cv2.cvtColor(cv2.resize(cv2.imread(file),dsize=(48,48)),cv2.COLOR_BGR2RGB)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":_float_feature(list(label)),
            "img":_byte_feature(img.tostring())
        }))
        write.write(example.SerializeToString())
# read
def showimg(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
def example6():
    reader = tf.TFRecordReader()
    import numpy as np
    filename = '../../tfrecord/record.tfrecords'
    filename_queue = tf.train.string_input_producer([filename])
    _,example, = reader.read(filename_queue)
    features = tf.parse_single_example(example,features={
        "label":tf.FixedLenFeature([10],tf.float32),
        "img":tf.FixedLenFeature([],tf.string)
    })
    imgs = tf.reshape(tf.decode_raw(features['img'],tf.uint8),shape=[48,48,3])
    label = tf.cast(features['label'],tf.float32)
    sess = tf.Session()
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess,coord)
    while True:
        i, l = sess.run([imgs, label])
        showimg(i)
        print(i,l)
def parse_single_exmp(serialized_example,labels_nums=2):
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
    tf_image =tf.reshape(tf_image, [48, 48, 3]) # 设置图像的维度
    return tf_image, tf_label

def example7():
    filename = '../../tfrecord/record.tfrecords'
    ds = tf.data.TFRecordDataset([filename]).map(parse_single_exmp).batch(300)
    iter_op = ds.make_initializable_iterator()
    init_op = iter_op.make_initializer(ds)
    data = iter_op.get_next()
    sess = tf.Session()
    sess.run(init_op)
    while True:
        d,l= sess.run(data)
        print(d.shape)
if __name__ == '__main__':
    example7()