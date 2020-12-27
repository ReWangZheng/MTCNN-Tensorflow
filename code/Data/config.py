"""
FDDB CONFIG
"""
FDDB_DIR='/home/dataset/FDDB/'
FDDB_DIR_FOLDS=FDDB_DIR+'/FDDB-folds/'

MAKE_SAVE='/home/dataset/FDDB/MTCNN_USED_DATA/'
POS_DIR = 'positive_face'
NEG_DIR = 'negtive_face'
PART_DIR = 'part_face'
def init_dir():
    import os
    if not os.path.exists(os.path.join(MAKE_SAVE,POS_DIR)):
        os.makedirs(os.path.join(MAKE_SAVE,POS_DIR))
    if not os.path.exists(os.path.join(MAKE_SAVE, NEG_DIR)):
        os.makedirs(os.path.join(MAKE_SAVE, NEG_DIR))
    if not os.path.exists(os.path.join(MAKE_SAVE, PART_DIR)):
        os.makedirs(os.path.join(MAKE_SAVE, PART_DIR))


"""CONFIG END"""

"""CONFIG AFLW"""
AFLW_DIR='/home/dataset/AFLW/'
AFLW_SQLIT='/home/dataset/AFLW/alfw/'