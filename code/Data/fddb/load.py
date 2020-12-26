import os
from tensorflow._api.v1.data import Dataset,Iterator
from sklearn.model_selection import train_test_split
import tensorflow as tf
class ImageDateSet:
    def __init__(self,positive_dir,negative_dir,img_size=(12,12),batch=50):
        #output image size
        self.img_size = img_size
        #output image bathch
        self.batch = batch
        # where the positive data from
        self.positive_dir = positive_dir
        # where the negative data from
        self.negative_dir = negative_dir
        # get the file list
        file_list = os.listdir(positive_dir)
        self.data_info = [(os.path.join(self.positive_dir,file_name),1) for file_name in file_list]
        # need to generate negative sample?
        if negative_dir is not None:
            file_list=os.listdir(negative_dir)
            self.data_info.extend([(os.path.join(self.negative_dir,fn),0) for fn in file_list])
        # split the train and test
        self.data_train,self.data_test = train_test_split(self.data_info)
        # make Dataset

        self.dataset_train = Dataset.from_tensor_slices(self.conver(self.data_train)).shuffle(500).map(self.load_img,num_parallel_calls=4).cache()
        self.dataset_train = self.dataset_train.batch(self.batch).repeat().prefetch(200)
        self.dataset_test = Dataset.from_tensor_slices(self.conver(self.data_test)).shuffle(500).map(self.load_img,num_parallel_calls=4).cache()
        self.dataset_test = self.dataset_test.batch(self.batch).repeat().prefetch(200)
    def getIterator(self):
        iter_train = self.dataset_train.make_initializable_iterator()
        train_op = iter_train.make_initializer(self.dataset_train)
        element_train = iter_train.get_next()

        iter_test = self.dataset_test.make_initializable_iterator()
        test_op = iter_test.make_initializer(self.dataset_test)
        element_test = iter_test.get_next()
        return train_op,element_train,test_op,element_test
    def conver(self,info):
        paths=[]
        labs=[]
        for path,lab in info:
            paths.append(path)
            labs.append(lab)
        return (paths,labs)
    def load_img(self,img_path,label):
        # read file
        img_file = tf.read_file(img_path)
        #decode the binary data to image
        img_decoded = tf.image.decode_jpeg(img_file, channels=3)
        # resize the image
        resized_image = tf.image.resize_images(img_decoded, [self.img_size[0], self.img_size[1]])
        # convert the label to one-hot encoding
        classes_num = 2
        clss = tf.one_hot(label, classes_num)
        return (resized_image,clss)
class BoundDataset:
    def __init__(self,info,img_size=(12,12),batch=50):
        #output image size
        self.img_size = img_size
        #output image bathch
        self.batch = batch
        # where the positive data from
        # get the file list
        file_paths = []
        bounds = []
        with open(info) as f:
            lines = f.readlines()
            for idx in range(0,len(lines),2):
                file_paths.append(lines[idx].strip())
                bounds.append([float(nub) for nub in lines[idx+1].strip().split(' ')])

        # split the train and test
        f1,f2,b1,b2 = train_test_split(file_paths,bounds)
        self.data_train = (f1,b1)
        self.data_test = (f2,b2)
        # make Dataset
        self.dataset_train = Dataset.from_tensor_slices(self.data_train).shuffle(800).map(self.load_img,num_parallel_calls=4).cache()
        self.dataset_train = self.dataset_train.batch(self.batch).repeat().prefetch(600)
        self.dataset_test = Dataset.from_tensor_slices(self.data_test).shuffle(800).map(self.load_img,num_parallel_calls=4).cache()
        self.dataset_test = self.dataset_test.batch(self.batch).repeat().prefetch(600)
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
def test_imagedata():
    ds = ImageDateSet(negative_dir='/home/dataset/FDDB/MTCNN_USED_DATA/negtive_face',positive_dir='/home/dataset/FDDB/MTCNN_USED_DATA/positive_face')
    train_op,element_train,test_op,element_test=ds.getIterator()
    sess = tf.Session()
    sess.run(train_op)
    print(sess.run(element_train))
def test_bounds():
    bd = BoundDataset(info='/home/dataset/FDDB/MTCNN_USED_DATA/part_info.txt')
    train_op, element_train, test_op, element_test = bd.getIterator()
    sess = tf.Session()
    sess.run(train_op)
    print(sess.run(element_train))
def make_TFrecard():
    classifier = ImageDateSet(negative_dir='/home/dataset/FDDB/MTCNN_USED_DATA/negtive_face',
                              positive_dir='/home/dataset/FDDB/MTCNN_USED_DATA/positive_face', batch=300)
    reg = BoundDataset(info='/home/dataset/FDDB/MTCNN_USED_DATA/part_info.txt', batch=300)
    c_trainop, c_train, c_testop, c_test = classifier.getIterator()
    r_trainop, r_train, r_testop, r_test = reg.getIterator()


if __name__ == '__main__':
    test_imagedata()