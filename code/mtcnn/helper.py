import tensorflow as tf
class ModelHelper:
    def __init__(self, save_path, sess: tf.Session, step=10):
        self.sess = sess
        self.saver = tf.train.Saver()
        self.every_step = step
        self.save_path = save_path
        self.log_opt = None
    def reload(self):
        if tf.train.checkpoint_exists(self.save_path):
            self.saver.restore(self.sess, self.save_path)
    def save(self):
        self.saver.save(self.sess, self.save_path)
class SummaryHelper:
    def __init__(self,namescope,logdir,graph):
        self.graph = graph
        self.scope = tf.name_scope(namescope)
        self.upt = None
        with self.graph.as_default():
            self.fw = tf.summary.FileWriter(logdir,graph)

    def add_scalar(self,scalar_dict:dict):
        with self.scope:
            for key in scalar_dict.keys():
                v = scalar_dict[key]
                tf.summary.scalar(key,v)
    @staticmethod
    def merge_all():
        return tf.summary.merge_all()
    def save(self,summary,step):
        self.fw.add_summary(summary,step)
