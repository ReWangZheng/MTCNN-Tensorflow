"""
FDDB CONFIG
"""
FDDB_DIR='/home/dataset/FDDB/'
FDDB_DIR_FOLDS=FDDB_DIR+'/FDDB-folds/'

FDDB_MAKE_SAVE='/home/dataset/FDDB/MTCNN_USED_DATA/'
POS_DIR = 'positive_face'
NEG_DIR = 'negtive_face'
PART_DIR = 'part_face'
def init_dir():
    import os
    if not os.path.exists(os.path.join(FDDB_MAKE_SAVE,POS_DIR)):
        os.makedirs(os.path.join(FDDB_MAKE_SAVE,POS_DIR))
    if not os.path.exists(os.path.join(FDDB_MAKE_SAVE, NEG_DIR)):
        os.makedirs(os.path.join(FDDB_MAKE_SAVE, NEG_DIR))
    if not os.path.exists(os.path.join(FDDB_MAKE_SAVE, PART_DIR)):
        os.makedirs(os.path.join(FDDB_MAKE_SAVE, PART_DIR))


"""CONFIG END"""

AFLW_DIR='/home/dataset/AFLW/'
