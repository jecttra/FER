import os

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
IMAGE_CROP_TOP = 12
IMAGE_CROP_BOTTOM = 60
IMAGE_CROP_LEFT = 90
IMAGE_CROP_RIGHT = 90

#Image Selections
peek = 0.6
neutral = 0.2

#Paths
source_path = '../common/cohn-kanade-images'
db_path = 'db'
snapshot_path = 'snapshots'

#Database
emotions = 'emotions.json'
train_lmdb = os.path.join(db_path,'train_lmdb')
test_lmdb = os.path.join(db_path,'test_lmdb')
train_binaryproto = os.path.join(db_path,'train_mean.binaryproto')
test_binaryproto = os.path.join(db_path,'test_mean.binaryproto')
train_npy = os.path.join(db_path,'train_mean.npy')
test_npy = os.path.join(db_path,'test_mean.npy')
train_json = os.path.join(db_path,'train.json')
test_json = os.path.join(db_path,'test.json')

#Train
solver_prototxt = 'solver.prototxt'
snapshot_prefix = snapshot_path + os.sep
pre_trained_weight = '../common/pre_trained/bvlc_reference_caffenet.caffemodel'

#Test
test_prototxt = 'train.prototxt'
final_layer = 'prob'
