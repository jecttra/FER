import caffe
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
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

def blobs_to_image(imname, net, layer, padding=5):
    if layer not in net.blobs:
        return
    data = net.blobs[layer].data[0]
    shape = data.shape
    if len(shape) != 3:
        return
    num_column = int(round(np.prod(shape)**(.5)/shape[2]))
    num_row = int(round(shape[0]/num_column + .5))
    height = num_row*(shape[1] + padding) - padding
    width = num_column*(shape[2] + padding) - padding
    result = np.zeros((height, width))
    x = 0
    y = 0
    for n in range(shape[0]):
        if x == num_row:
            x = 0
            y += 1
        a = y * (shape[1] + padding)
        b = x * (shape[2] + padding)
        for j in range(shape[2]):
            for i in range(shape[1]):
                result[a + i, b + j] = data[n, i, j]
        x += 1
    plt.imsave(imname, result, cmap='gray')

def test():
    test_net = caffe.Net(test_prototxt, snapshot_prefix + '_iter_%d.caffemodel' % start_iter, caffe.TEST)
    transformer = caffe.io.Transformer({'data': test_net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(test_npy).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    length = test_net.blobs[final_layer].data.shape[1]
    top_k = [0] * length
    size = 0

    import json
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    for s, item in test_data.items():
        for n, ds in item.items():
            for d in ds:
                expected = d[1]
                d = d[0]
                fname = os.path.join(source_path, '{s}/{n}/{s}_{n}_{d:08d}.png'.format(e=expected, s=s, n=n, d=d))
                image = caffe.io.load_image(fname)
                #Crop
                image = image[IMAGE_CROP_TOP:-IMAGE_CROP_BOTTOM-1, IMAGE_CROP_LEFT:-IMAGE_CROP_RIGHT-1]
                image = transformer.preprocess('data', image)

                #Forward
                test_net.forward(data=np.asarray([image]))

                #Get results
                result = test_net.blobs[final_layer].data[0].flatten().argsort()[::-1].tolist()

                # Uncomment to output intermediate images
                #img_prefix = 'layers/{s}_{n}_{d:08d}/'.format(s=s, n=n, d=d)
                #os.makedirs(img_prefix, exist_ok=True)
                #for layer in test_net.blobs.keys():
                #    blobs_to_image(img_prefix + layer + '.png', test_net, layer)

                #Calculate Accuracy
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
        print ("using pre_trained:", pre_trained_weight)
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
    train(pre_trained_weight)
else:
    test()
