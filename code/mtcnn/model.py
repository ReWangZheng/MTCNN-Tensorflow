from tqdm import tqdm
import sys
from mtcnn.helper import ModelHelper,SummaryHelper
from mtcnn.factory import *
from Data.DataSet import TrainDateSet
flags = tf.flags.FLAGS
tf.flags.DEFINE_float('lr',default=0.01,help='学习率的初始值')

class MTCNNNet:
    def __init__(self, net, is_train=True):
        self.graph = tf.Graph()
        net_factory = None
        loss_rate = []
        if net == 'pnet':
            net_factory = Pnet_factory
            loss_rate = [1., 0.5, 0.5]
        elif net == 'rnet':
            net_factory = Rnet_factory
            loss_rate = [1., 0.5, 0.5]
        elif net == 'onet':
            net_factory = Onet_factory
            loss_rate = [1., 0.5, 1.]
        with self.graph.as_default():
            self.x_input, self.face_c, self.bound_r, self.landmakr_r = net_factory()
            self.y_label = tf.placeholder(dtype=tf.float32, shape=[None, 2])
            self.y_bound = tf.placeholder(dtype=tf.float32, shape=[None, 4])
            self.y_landmark = tf.placeholder(dtype=tf.float32, shape=[None, 10])
            self.label = tf.placeholder(dtype=tf.float32, shape=[None, ])
            self.acc, self.cls_prob, self.cls_loss = Classifier_loss_factory(self.label, self.y_label, self.face_c)
            self.bound_loss = Bounding_box_loss_factory(self.label, self.bound_r, self.y_bound)
            self.landmark_loss = Landmark_loss_factory(self.label, self.landmakr_r, self.y_landmark)
            self.com_loss = self.cls_loss * loss_rate[0] + self.bound_loss * loss_rate[1] + self.landmark_loss * loss_rate[2]
            if is_train:
                self.global_step = tf.Variable(0)
                self.add_step = tf.assign_add(self.global_step, 1)
                self.lr = tf.train.exponential_decay(learning_rate=flags.lr, global_step=self.global_step, decay_steps=1000,
                                                     decay_rate=0.5)
                self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.com_loss)
            self.sess = tf.Session()
            self.model_saver = ModelHelper(save_path='../../model/{}/{}.ckpt'.format(net,net), sess=self.sess)
            self.dataset = TrainDateSet(self.sess,Net=net)
            self.summary = SummaryHelper(logdir='../../logdir/',namescope=net,graph=self.graph)
            self.summary.add_scalar({'learning_rate':self.lr,
                                     'bound_loss':self.bound_loss,
                                     'landmark_loss':self.landmark_loss,
                                     "combination_loss":self.com_loss})
            self.summary_opt = self.summary.merge_all()
    def train(self):
        with self.graph.as_default():
            # init
            self.sess.run(tf.global_variables_initializer())
            self.dataset.init()
            self.model_saver.reload()
            for i in range(0, 400):
                for step in tqdm(range(0, 20), file=sys.stdout):
                    X, YC, YR, YL, label = self.dataset.get_data()
                    self.sess.run([self.face_c, self.train_opt, self.add_step],
                                  feed_dict={self.x_input: X, self.y_label: YC, self.y_bound: YR, self.label: label,
                                             self.y_landmark: YL})
                X, YC, YR, YL, label = self.dataset.get_data()
                run_task = [self.landmark_loss, self.cls_prob, self.global_step, self.acc, self.cls_loss, self.bound_loss,self.com_loss,self.summary_opt]
                feed_task = {self.y_landmark: YL,self.x_input: X, self.y_label: YC, self.y_bound: YR,self.label: label}
                ld_loss, prob, g_step, acc, closs, bloss, comloss,summary = self.sess.run(run_task,feed_task)
                self.model_saver.save()
                self.summary.save(summary,g_step)
                print('step {},classifier loss:{:.4} acc {:.4},bound loss:{:.4},landmark loss:{:.4},com loss:{:.4}'.format(
                    g_step, closs, acc, bloss, ld_loss, comloss))
    def predict(self, img):
        if img.ndim == 3:
            img = np.reshape(img, [1, *img.shape])
        with self.graph.as_default():
            self.model_saver.reload()
            prob = self.sess.run(self.cls_prob, feed_dict={self.x_input: img, self.label: np.ones(shape=[len(img)])})
            br = self.sess.run(self.bound_r, feed_dict={self.x_input: img, self.label: np.ones(shape=[len(img)])})
            lm = self.sess.run(self.landmakr_r, feed_dict={self.x_input: img, self.label: np.ones(shape=[len(img)])})
            return prob, br, lm
class MTCNN:
    def __init__(self):
        pass
if __name__ == '__main__':
    MTCNNNet(net='rnet').train()