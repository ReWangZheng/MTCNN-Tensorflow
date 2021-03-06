import os
import sys
from tqdm import tqdm
import random
from util import *
"""
FDDB CONFIG
"""
FDDB_DIR='/home/dataset/FDDB/'
FDDB_DIR_FOLDS=FDDB_DIR+'/FDDB-folds/'
MAKE_SAVE='/home/dataset/FDDB/MTCNN_USED_DATA/'
POS_DIR = 'positive_face'
NEG_DIR = 'negtive_face'
PART_DIR = 'part_face'
data_info = \
    ['FDDB-fold-05-ellipseList.txt',
     'FDDB-fold-01-ellipseList.txt',
     'FDDB-fold-04-ellipseList.txt',
     'FDDB-fold-06-ellipseList.txt',
     'FDDB-fold-02-ellipseList.txt',
     'FDDB-fold-03-ellipseList.txt',
     'FDDB-fold-09-ellipseList.txt',
     'FDDB-fold-10-ellipseList.txt',
     'FDDB-fold-07-ellipseList.txt',
     'FDDB-fold-08-ellipseList.txt']
"""CONFIG END"""
def init_dir():
    import os
    if not os.path.exists(os.path.join(MAKE_SAVE,POS_DIR)):
        os.makedirs(os.path.join(MAKE_SAVE,POS_DIR))
    if not os.path.exists(os.path.join(MAKE_SAVE, NEG_DIR)):
        os.makedirs(os.path.join(MAKE_SAVE, NEG_DIR))
    if not os.path.exists(os.path.join(MAKE_SAVE, PART_DIR)):
        os.makedirs(os.path.join(MAKE_SAVE, PART_DIR))
def read_fddb_base():
    fddb_info = []
    for txt in data_info:
        file = os.path.join(FDDB_DIR_FOLDS, txt)
        with open(file) as f:
            mes_list = [mes.strip() for mes in f.readlines()]
            idx = 0
            while idx < len(mes_list):
                filename = os.path.join(FDDB_DIR, mes_list[idx] + '.jpg')
                idx += 1
                face_size = int(mes_list[idx])
                idx += 1
                face_box = []
                for i in range(0, face_size):
                    l, s, _, x, y, _, _ = mes_list[idx].split(' ')
                    x_min = float(x) - float(s)
                    y_min = float(y) - float(l)
                    W = float(s) * 2
                    H = float(l) * 2
                    face_box.append([max(x_min,0), max(y_min,0), W, H])
                    idx += 1
                fddb_info.append((filename, face_size, face_box))
    return fddb_info
def make():
    init_dir()
    def random_offset():
        x_t = (np.random.rand() % 0.5) * random.sample([-1, 1], 1)[0]
        y_t = (np.random.rand() % 0.5) * random.sample([-1, 1], 1)[0]
        w_t = (0.5 + np.random.rand() * 1.5)
        h_t = (0.5 + np.random.rand() * 1.5)
        return x_t,y_t,w_t,h_t
    def extract(image:np.ndarray,ground_truths):
        h,w,c = image.shape
        positive_face = []
        part_face = []
        part_face_bound = []
        negtive_face=[]
        # create pos img
        sign = [(1,1),(1,-1),()]
        for idx,ground_truth in enumerate(ground_truths):
            x_p,y_p,w_p,h_p = np.array(ground_truth,int)
            true_img = image[y_p:y_p+h_p,x_p:x_p+w_p]
            positive_face.append(cv2.flip(true_img, 1))
            positive_face.append(cv2.flip(true_img, 0))
            positive_face.append(cv2.flip(true_img, -1))
            part_face.append(true_img)
            part_face_bound.append([0, 0, 1,1])
            # create part img
            for i in range(1,6):
                x_t,y_t,w_t,h_t = random_offset()
                x_of = max(int(x_p + w_p * x_t),0)
                y_of = max(int(y_p + h_p * y_t),0)
                w_of = min(int(w_p * w_t),w_p)
                h_of = min(int(h_p * h_t),h_p)
                part_face.append(image[y_of:y_of+h_of,x_of:x_of+w_of])
                part_face_bound.append([-x_t,-y_t,1/w_t,1/h_t])
        # create neg img
        save = True
        for i in range(12):
            x_n = np.random.randint(0, w-48)
            y_n = np.random.randint(0, h-48)
            w_n = np.random.randint(48, w-x_n)
            h_n = np.random.randint(48, h-y_n)
            if w_n==0 or h_n==0:
                continue
            for true_box in ground_truths:
                _,_,iou=overlap(true_box,[x_n,y_n,w_n,h_n])
                if iou>0.4:
                    save = False
                    break
            if save:
                negtive_face.append(image[y_n:y_n+h_n,x_n:x_n+w_n])
            save=True
        return positive_face,(part_face,part_face_bound),negtive_face
    data_infomation = read_fddb_base()
    save_dir = MAKE_SAVE
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    parts_img = []
    parts_label = []
    idx = 0
    f = open(os.path.join(MAKE_SAVE,'part_info.txt'),mode='w')
    for item_info in tqdm(data_infomation,file=sys.stdout,desc='Making neg/pos Set '):
        img_path,face_number,boxs = item_info
        img_origin = cv2.imread(img_path)
        pos,part,neg = extract(img_origin,boxs)
        save_images(neg,os.path.join(MAKE_SAVE,NEG_DIR))
        save_images(pos,os.path.join(MAKE_SAVE,POS_DIR))
        names = save_images(part[0], os.path.join(MAKE_SAVE, PART_DIR))
        for name,t in zip(names,part[1]):
            f.write('{}\n{} {} {} {}\n'.format(name,t[0],t[1],t[2],t[3]))
if __name__ == '__main__':
    make()