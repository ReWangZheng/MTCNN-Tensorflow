import Data
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
def save_images(imgs,save_dir,names=None):
    import cv2
    import os
    from uuid import uuid4
    if names is None:
        names = [os.path.join(save_dir,'{}.jpg'.format(uuid4())) for i in range(0,len(imgs))]
    for i in range(0,len(imgs)):
        cv2.imwrite(names[i],imgs[i])
    return names