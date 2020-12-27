# MTCNN-Tensorflow
基于Tensorflow的MTCNN实现，可用于实时的人脸检测

### 处理aflw数据集

第一步，更改Data.aflw.maker中的配置

    sub_set0、sub_set2:aflw三个数据文件夹中的一个

    info_dir:制作好的数据集标签将要存储的位置

    pos_dir:分割出来的人脸存储的位置

    sqlite_path: aflw数据集中带有的sqlite文件路径

第二步，更改好路径之后，直接先调用`Make_info()`，然后在调用`do_make()`
即可