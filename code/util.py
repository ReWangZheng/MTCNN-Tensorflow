import numpy as np
import cv2
def drwa_bbox(img_origin,bbox,color=(255,0,0)):
    img = np.copy(img_origin)
    for box in bbox:
        h,w,_ = img.shape
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = x1+int(box[2] * w)
        y2 = y1+int(box[3] * h)
        img = cv2.rectangle(img,(x1,y1),(x2,y2),color=color,thickness=2)
    return img
def NMS(bboxs,probs,thread=0.5):
    '''
    :param bboxs: [[x,y,w,h].....]
    :param probs [p1,p2,p3....]
    :return:
    '''
    index_box = np.argsort(np.array(probs) * -1)
    res_box= []

    used_box = [False for i in range(0,len(probs))]
    res_id = []
    for bidx in index_box:
        box_prop_max = bboxs[bidx]
        if used_box[bidx]:
            continue
        res_box.append(box_prop_max)
        res_id.append(bidx)
        used_box[bidx] = True
        for idx,box in enumerate(bboxs):
            if used_box[idx]:
                continue
            iou,box1_iou,box2_iou = overlap(box_prop_max,box)
            if box2_iou > thread or box1_iou>thread:
                used_box[idx] = True
    return res_box,res_id

def img_pyramids(img,pyramcount=3,winsize=(48,48),step=(10,10)):
    def slide_window(img, stride=[5, 5], win=[48, 48]):
        h, w = img.shape[:2]
        for y in range(0, h, stride[1]):
            for x in range(0, w, stride[0]):
                window = img[y:y + win[1], x:x + win[0]]
                yield window, [x / w, y / h, (x + win[0]) / w, (y + win[1]) / h]
    pyramids = [np.copy(img)]
    imgs_win = []
    bbox = []
    for i in range(1,pyramcount):
        pyr_img = cv2.pyrDown(pyramids[i-1])
        pyramids.append(pyr_img)
    for image in pyramids:
        for img_win,box in slide_window(image,stride=step,win=winsize):
            if img_win.shape[0] != winsize[0] or img_win.shape[1] != winsize[1]:
                continue
            imgs_win.append(img_win)
            bbox.append(box)
    return pyramids,imgs_win,bbox
def overlap(box1,box2):
    in_x_min = max(box1[0],box2[0])
    in_y_min = max(box1[1],box2[1])
    in_x_max = min(box1[0]+box1[2],box2[0]+box2[2])
    in_y_max = min(box1[1]+box1[3],box2[1]+box2[3])
    if in_x_min>in_x_max or in_y_min>in_y_max:
        return 0.0,0.0,0.0
    w_in, h_in = in_x_max-in_x_min,in_y_max-in_y_min
    _, _, w_1, h_1 = box1
    _, _, w_2, h_2 = box2
    # compute the boxs' area
    in_area = w_in * h_in
    box1_area = w_1 * h_1
    box2_area = w_2 * h_2
    iou = in_area / float(box1_area+box2_area)
    box1_iou = in_area / float(box1_area)
    box2_iou = in_area / float(box2_area)
    return iou,box1_iou,box2_iou
