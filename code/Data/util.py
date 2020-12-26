import Data
def save_images(imgs,save_dir,names=None):
    import cv2
    import os
    from uuid import uuid4
    if names is None:
        names = [os.path.join(save_dir,'{}.jpg'.format(uuid4())) for i in range(0,len(imgs))]
    for i in range(0,len(imgs)):
        cv2.imwrite(names[i],imgs[i])
    return names