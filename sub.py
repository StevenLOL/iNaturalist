import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import sys
import numpy as np
import cv2
import json
from common import find_mxnet
import mxnet as mx

def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs

def oversample(images, crop_dims):

    im_shape = np.array(images.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # print crops_ix

    # Extract crops
    crops = np.empty((10, crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    # for im in images:
    im = images
    # print im.shape
    for crop in crops_ix:
        # print crop
        crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
        # cv2.imshow('crop', im[crop[0]:crop[2], crop[1]:crop[3], :])
        # cv2.waitKey()
        ix += 1
    crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]
    # cv2.imshow('crop', crops[0,:,:,:])
    # cv2.waitKey()
    return crops

#prefix = 'model/Scence-resnet-152-365'
prefix = 'model/Scence-resnet-152'
epoch = int(sys.argv[1]) #check point step
gpu_id = int(sys.argv[2]) #GPU ID for infer
ctx = mx.gpu(gpu_id)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)


ann_file = 'data/scene_validation_20170908.json'
print('Loading annotations from: ' + os.path.basename(ann_file))
with open(ann_file) as data_file:
    ann_data = json.load(data_file)

imgs = [aa['image_id'] for aa in ann_data]

classes = [0]*len(imgs)

#idx_to_class = {cc['id']: cc['name'] for cc in ann_data['categories']}

IMAGE_DIR="data/scene_validation_20170908/"

top1_acc = 0
top5_acc = 0
cnt = 0
img_sz = 450
crop_sz = 400

result = []

#for index in range(batch_sz*num_batches, len(imgs)):
for index in range(0, len(imgs)):
        img_name = imgs[index]
        label = str(classes[index])

        cnt += 1
        img_full_name = IMAGE_DIR + img_name
        img = cv2.cvtColor(cv2.imread(img_full_name), cv2.COLOR_BGR2RGB)
        img = np.float32(img)

        rows, cols = img.shape[:2]
        if cols < rows:
            resize_width = img_sz
            resize_height = resize_width * rows / cols;
        else:
            resize_height = img_sz
            resize_width = resize_height * cols / rows;

        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)

        #batch = oversample(img, (crop_sz,crop_sz))

        h, w, _ = img.shape

        x0 = int((w - crop_sz) / 2)
        y0 = int((h - crop_sz) / 2)
        img = img[y0:y0+crop_sz, x0:x0+crop_sz]

        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # change to r,g,b order

        img = img[np.newaxis, :]
        arg_params["data"] = mx.nd.array(img, ctx)
        #arg_params["data"] = mx.nd.array(input_blob, ctx)
        arg_params["softmax_label"] = mx.nd.empty((1,), ctx)
        exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
        exe.forward(is_train=False)
        probs = exe.outputs[0].asnumpy()
        score = np.squeeze(probs.mean(axis=0))

        sort_index = np.argsort(score)[::-1]
        top_k = sort_index[0:3]
	#print(top_k)

        temp_dict = {}

        temp_dict['label_id'] = top_k.tolist()
        temp_dict['image_id'] = img_name
        result.append(temp_dict)


with open('submit.json', 'w') as f:
    json.dump(result, f)
    print('write result json, num is %d' % len(result))




