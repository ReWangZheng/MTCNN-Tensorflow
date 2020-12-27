import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow._api.v1.data import Dataset,Iterator
data_info_path = '/home/dataset/AFLW/MTCNN/data_info.csv'

class LandMarkData:
    def __init__(self,info=data_info_path,img_size=(48,48),batch=50):
        #output image size
        self.img_size = img_size
        #output image bathch
        self.batch = batch
        # where the positive data from
        # get the file list
        df = pd.read_csv(info)
        file_list = df.iloc[:,1].to_numpy()
        land_mark = df.iloc[:,2:].to_numpy()
        # split the train and test
        f1,f2,b1,b2 = train_test_split(file_list,land_mark)
        self.data_train = (f1,b1)
        self.data_test = (f2,b2)
        # make Dataset
        self.dataset_train = Dataset.from_tensor_slices(self.data_train).shuffle(150).map(self.load_img,num_parallel_calls=4).cache()
        self.dataset_train = self.dataset_train.batch(self.batch).repeat().prefetch(400)
        self.dataset_test = Dataset.from_tensor_slices(self.data_test).shuffle(150).map(self.load_img,num_parallel_calls=4).cache()
        self.dataset_test = self.dataset_test.batch(self.batch).repeat().prefetch(400)
    def getIterator(self):
        iter_train = self.dataset_train.make_initializable_iterator()
        train_op = iter_train.make_initializer(self.dataset_train)
        element_train = iter_train.get_next()

        iter_test = self.dataset_test.make_initializable_iterator()
        test_op = iter_test.make_initializer(self.dataset_test)
        element_test = iter_test.get_next()
        return train_op,element_train,test_op,element_test
    def load_img(self,img_path,label):
        # read file
        img_file = tf.read_file(img_path)
        #decode the binary data to image
        img_decoded = tf.image.decode_jpeg(img_file, channels=3)
        # resize the image
        resized_image = tf.image.resize_images(img_decoded, [self.img_size[0], self.img_size[1]])
        return resized_image,label

def read_data_test():
    df = pd.read_csv(data_info_path)
    file_list = df.iloc[:, 1].to_numpy()
    land_mark = df.iloc[:, 2:].to_numpy()
    f = tf.train.slice_input_producer((file_list,land_mark),capacity=1000)
    i=0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        while True:
            k = sess.run(f)
            i+=1
            print(i)


if __name__ == '__main__':
    read_data_test()
