import os
import glob
import random
import numpy as np
import os
import sys
import cv2

import matplotlib
import caffe
from caffe.proto import caffe_pb2
import lmdb
import json
sys.path.append('.')
from vars import *

with open(emotions, 'r') as f:
    emotions = json.load(f)

data = {}

def get_tup(root, name, emotion=True):
    parts = name.split('_')
    s = parts[0]
    n = parts[1]
    path = os.path.join(root, name)
    d = int(parts[2].split('.')[0])
    e = 0
    if emotion:
        e = emotions[s][n]
    return (path, e, s, n, d)

size = 0

for root, dirs, files in os.walk(source_path):
    if len(files) == 0:
        continue
    parts = root.split(os.sep)
    s = parts[-2]
    n = parts[-1]
    if s not in emotions or n not in emotions[s]:
        continue
    if root not in data:
        data[root] = []
    files = [f for f in sorted(files) if f != '.DS_Store']
    for name in files[int(len(files) * peek):]:
        data[root].append(get_tup(root, name))
        size += 1
    for name in files[:int(len(files) * neutral)]:
        data[root].append(get_tup(root, name, False))
        size += 1

in_data = list(data.items())
random.shuffle(in_data)
data = []
for root_idx, (root, tups) in enumerate(in_data):
    for tup in tups:
        data.append((root_idx % 8 == 0,) + tup)
random.shuffle(data)

os.makedirs(db_path, exist_ok=True)
os.makedirs(snapshot_path, exist_ok=True)
os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + test_lmdb)

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Crop
    img = img[IMAGE_CROP_TOP:-IMAGE_CROP_BOTTOM-1, IMAGE_CROP_LEFT:-IMAGE_CROP_RIGHT-1]

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    #cv2.imwrite('test.png',img)
    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

train_dict = {}
test_dict = {}

train_in_db = lmdb.open(train_lmdb, map_size=int(size * 3 * IMAGE_WIDTH * IMAGE_HEIGHT * 2))
train_in_txn = train_in_db.begin(write=True)
test_in_db = lmdb.open(test_lmdb, map_size=int(size * 3 * IMAGE_WIDTH * IMAGE_HEIGHT * 2))
test_in_txn = test_in_db.begin(write=True)
in_idx = 0
for in_idx, (is_test, img_path, label, s, n, d) in enumerate(data):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    datum = make_datum(img, label).SerializeToString()
    text = '{:0>5d}'.format(in_idx).encode('ascii')
    if is_test:
        if s not in test_dict:
            test_dict[s] = {}
        if n not in test_dict[s]:
            test_dict[s][n] = []
        test_dict[s][n].append((d, label))
        test_in_txn.put(text, datum)
    else:
        if s not in train_dict:
            train_dict[s] = {}
        if n not in train_dict[s]:
            train_dict[s][n] = []
        train_dict[s][n].append((d, label))
        train_in_txn.put(text, datum)
    print(text.decode('ascii') + ':' + img_path + ':' + str(label))
train_in_txn.commit()
test_in_txn.commit()
print(train_in_db.stat())
print(test_in_db.stat())
train_in_db.close()
test_in_db.close()

with open(train_json, 'w') as f:
    json.dump(train_dict, f, indent=2, sort_keys=True)
with open(test_json, 'w') as f:
    json.dump(test_dict, f, indent=2, sort_keys=True)

from subprocess import call
call(["compute_image_mean", train_lmdb, train_binaryproto])
call(["compute_image_mean", test_lmdb, test_binaryproto])

def to_npy(binaryproto, npy):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(binaryproto , 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    np.save(npy, arr[0])

to_npy(train_binaryproto, train_npy)
to_npy(test_binaryproto, test_npy)
