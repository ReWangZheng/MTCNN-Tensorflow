import sqlite3
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
import cv2
from ast import literal_eval

sub_set0 = '/home/dataset/AFLW/aflw/data/flickr/0'
sub_set2 = '/home/dataset/AFLW/aflw/data/flickr/2'
info_dir = '/home/dataset/AFLW/aflw/data/flickr/'

data_dir ='/home/dataset/AFLW/MTCNN/'
pos_dir = '/home/dataset/AFLW/MTCNN/pos'

sqlite_path = '/home/dataset/AFLW/aflw/data/aflw.sqlite'
need_land_mark_index = (8,11,15,18,20)
con = sqlite3.connect(sqlite_path)

def Make_info():
    maker_info(sub_set0,os.path.join(info_dir,'info_set0.csv'))
    maker_info(sub_set2,os.path.join(info_dir,'info_set2.csv'))
def maker_info(set_dir,save_name):
    file_list = os.listdir(set_dir)
    print(len(file_list))
    coloms = ['file_name']
    data_frame = []
    for file_name in tqdm(file_list,file=sys.stdout):
        line = []
        file_path = os.path.join(set_dir,file_name)
        faceid=get_faceid(file_name)
        if faceid is None:
            continue
        # add face id
        line.append(faceid)
        # add file path
        line.append(file_path)
        # add rect
        line.append(list(get_rect(faceid)))
        # get landmark
        land_mark = get_landmark(faceid)
        lm_dic = {8:[0,0],11:[0,0],15:[0,0],18:[0,0],20:[0,0]}
        for lm in land_mark:
            lm_dic[lm[0]] = list(lm[1:])
        lm_list = []
        for p in lm_dic.values():
            lm_list.extend(p)
        line.append(lm_list)
        data_frame.append(line)
    df=pd.DataFrame(data_frame)
    df.to_csv(save_name)
def read_fromsqlite():
    c = con.cursor()
    res = c.execute("select name from sqlite_master where type = 'table' order by name")
    print(res.fetchall())
    res2 = c.execute("select * from FaceImages")
    print(res2.fetchmany(50))
def get_faceid(file_name):
    c = con.cursor()
    res = c.execute("select face_id from Faces where file_id=?",[file_name])
    r = res.fetchone()
    if r is None:
        return None
    return r[0]
def get_rect(face_id):
    c = con.cursor()
    res = c.execute("select x,y,w,h from FaceRect where face_id=?", [face_id])
    return res.fetchone()
def get_landmark(face_id):
    c = con.cursor()
    res = c.execute("select feature_id,x,y from FeatureCoords where feature_id in {} and face_id={}".format(str(need_land_mark_index),face_id))
    return res.fetchall()
def do_make(positive_dir,infos,save_info):
    from uuid import uuid4
    data_infos = []
    for info in infos:
        data_infos.append(pd.read_csv(info))
    df = pd.concat(data_infos)
    frame = []
    for idx in tqdm(range(df.shape[0]),file=sys.stdout):
        line=[]
        file_path, rect, landmark = df.iloc[idx,2:]
        rect = literal_eval(rect)
        if rect[0]<0 or rect[1]<0 or rect[2]<=0 or rect[3]<=0:
            continue
        landmark = literal_eval(landmark)
        img = cv2.imread(file_path)
        # img = cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),color=(0,0,255),thickness=2)
        face_img = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        if len(face_img)==0:
            continue
        lm_list = []
        for i in range(0,len(landmark),2):
            x1 = int(landmark[i])
            y1 = int(landmark[i+1])
            if x1 !=0 and y1 !=0:
                x_n,y_n=(x1-rect[0])/rect[2],(y1-rect[1])/rect[3]
                lm_list.append(x_n)
                lm_list.append(y_n)
            else:
                lm_list.append(0.)
                lm_list.append(0.)
        fp = os.path.join(positive_dir,'{}.jpg'.format(uuid4()))
        line.append(fp)
        line.extend(lm_list)
        cv2.imwrite(fp,face_img)
        frame.append(line)
    new_info = pd.DataFrame(frame)
    new_info.to_csv(save_info)
if __name__ == '__main__':
    do_make(pos_dir,[os.path.join(info_dir,'info_set0.csv'),os.path.join(info_dir,'info_set2.csv')],os.path.join(data_dir,'data_info.csv'))