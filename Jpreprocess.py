
import os
import numpy as np
import pickle
import nibabel as nib

modalities = ('flair', 't1ce', 't1', 't2')

# train
train_set = {
        'root': 'MICCAI_BraTS_2019_Data_Training/Train',
        'file_list': 'train.txt',
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the nii.gz file!')
    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def process_pkl_ui8f32b0(path):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """

    label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')
    images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)  # [240,240,155]


    mask = images.sum(-1) > 0
    for k in range(4):

        x = images[..., k]
        y = x[mask]

        x[mask] -= y.mean()
        x[mask] /= y.std()
        images[..., k] = x

    output = path + 'pkl_ui8f32b0.pkl'
    with open(output, 'wb') as f:
        print(output)
        pickle.dump((images, label), f)


def topkl(datasets_set):
    root = datasets_set['root']
    file_list = os.path.join(root, datasets_set['file_list'])
    subjects = open(file_list).read().splitlines()
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]

    for path in paths:
        process_pkl_ui8f32b0(path)

if __name__ == '__main__':
    topkl(train_set)