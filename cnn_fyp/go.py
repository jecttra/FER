import caffe
import sys
import numpy as np
import cv2
sys.path.append('.')
from vars import *

# python3 go.py train|test [iteration]

caffe.set_mode_gpu()

mode = sys.argv[1] #train or test
start_iter = 0
if len(sys.argv) > 2:
    start_iter = int(sys.argv[2])
else:
    for root, dirs, files in os.walk(snapshot_path):
        for name in files:
            i = int(name.split('.')[0].split('_')[-1])
            if i > start_iter:
                start_iter = i

def test():
    test_net = caffe.Net(test_prototxt, snapshot_prefix + '_iter_%d.caffemodel' % start_iter, caffe.TEST)
    transformer = caffe.io.Transformer({'data': test_net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(test_npy).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    def test_image(fname, i, net):
        image = caffe.io.load_image(fname)
        #Crop
        image = image[IMAGE_CROP_TOP:-IMAGE_CROP_BOTTOM-1, IMAGE_CROP_LEFT:-IMAGE_CROP_RIGHT-1]
        image = transformer.preprocess('data', image)
        out = net.forward(data=np.asarray([image]))
        return net.blobs[final_layer].data[0].flatten().argsort()[::-1].tolist()
    length = test_net.blobs[final_layer].data.shape[1]
    top_k = [0] * length
    size = 0

    import json
    import os
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    for s, item in test_data.items():
        for n, ds in item.items():
            for d in ds:
                expected = d[1]
                d = d[0]
                result = test_image(os.path.join(source_path, '{s}/{n}/{s}_{n}_{d:08d}.png'.format(e=expected, s=s, n=n, d=d)), start_iter, test_net)
                index = result.index(expected)
                for i in range(index, length):
                    top_k[i] += 1
                size += 1
                print((s, n, d, expected, result, index))

    print('Top k: ' + str([i/size for i in top_k]))



def train(pre_trained_weight=None):
    solver = caffe.SGDSolver(solver_prototxt)
    if start_iter > 0:
        solver.restore(snapshot_prefix + '_iter_%d.solverstate' % start_iter)
    elif pre_trained_weight is not None:
        solver.net.copy_from(pre_trained_weight)
    curr_iter = start_iter
    try:
        while 1:
            solver.step(1)
            curr_iter += 1
    except KeyboardInterrupt:
        solver.snapshot()
        raise


if mode == 'train':
    train()
elif mode == 'trainP':
    train("../common/pre_trained/bvlc_reference_caffenet.caffemodel")
else:
    test()
