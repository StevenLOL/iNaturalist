# iNatularist image loader


from PIL import Image
import os
import json
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

def gen_list(prefix):
    ann_file = '%s.json'%prefix
    train_out = '%s.lst'%prefix
    # load annotations
    print('Loading annotations from: ' + os.path.basename(ann_file))
    with open(ann_file) as data_file:
        ann_data = json.load(data_file)

    # set up the filenames and annotations
    imgs = [prefix+ "/" +aa['image_id'] for aa in ann_data]
    im_ids = [aa['label_id'] for aa in ann_data]

    print('\t' + str(len(imgs)) + ' images')

    import pandas as pd
    from sklearn.utils import shuffle

    df = pd.DataFrame(im_ids)
    df[1] = imgs
    df = shuffle(df)

    df.to_csv(train_out, sep='\t', header=None, index=False)
    df = pd.read_csv(train_out, delimiter='\t', header=None)
    df.to_csv(train_out, sep='\t', header=None)

if __name__ == '__main__':
    set_names = ['scene_train_20170904', 'scene_validation_20170908']
    for name in set_names:
        gen_list(name)
