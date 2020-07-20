import os
import torch
import pickle
from PIL import Image


def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


data_train_train = load_data('MiniImagenet/miniImageNet_category_split_train_phase_train.pickle')
data_train_val = load_data('MiniImagenet/miniImageNet_category_split_train_phase_val.pickle')
data_train_test = load_data('MiniImagenet/miniImageNet_category_split_train_phase_test.pickle')

data_val = load_data('MiniImagenet/miniImageNet_category_split_val.pickle')
data_test = load_data('MiniImagenet/miniImageNet_category_split_test.pickle')

print('train_train')
folder = 'train_train'
os.mkdir(folder)
for key, value in data_train_train['catname2label'].items():
    os.mkdir(os.path.join(folder, key))
for i in range(len(data_train_train['labels'])):
    img = Image.fromarray(data_train_train['data'][i])
    cls = list(data_train_train['catname2label'].keys())[list(data_train_train['catname2label'].values()).index(data_train_train['labels'][i])]
    img_folder = os.path.join(folder, cls)
    img_name = os.path.join(img_folder, '{:05d}.png'.format(i))
    img.save(img_name)

print('train_val')
folder = 'train_val'
os.mkdir(folder)
for key, value in data_train_val['catname2label'].items():
    os.mkdir(os.path.join(folder, key))
for i in range(len(data_train_val['labels'])):
    img = Image.fromarray(data_train_val['data'][i])
    cls = list(data_train_val['catname2label'].keys())[list(data_train_val['catname2label'].values()).index(data_train_val['labels'][i])]
    img_folder = os.path.join(folder, cls)
    img_name = os.path.join(img_folder, '{:05d}.png'.format(i))
    img.save(img_name)

print('train_test')
folder = 'train_test'
os.mkdir(folder)
for key, value in data_train_test['catname2label'].items():
    os.mkdir(os.path.join(folder, key))
for i in range(len(data_train_test['labels'])):
    img = Image.fromarray(data_train_test['data'][i])
    cls = list(data_train_test['catname2label'].keys())[list(data_train_test['catname2label'].values()).index(data_train_test['labels'][i])]
    img_folder = os.path.join(folder, cls)
    img_name = os.path.join(img_folder, '{:05d}.png'.format(i))
    img.save(img_name)

print('val')
folder = 'val'
os.mkdir(folder)
for key, value in data_val['catname2label'].items():
    os.mkdir(os.path.join(folder, key))
for i in range(len(data_val['labels'])):
    img = Image.fromarray(data_val['data'][i])
    cls = list(data_val['catname2label'].keys())[list(data_val['catname2label'].values()).index(data_val['labels'][i])]
    img_folder = os.path.join(folder, cls)
    img_name = os.path.join(img_folder, '{:05d}.png'.format(i))
    img.save(img_name)

print('test')
folder = 'test'
os.mkdir(folder)
for key, value in data_test['catname2label'].items():
    os.mkdir(os.path.join(folder, key))
for i in range(len(data_test['labels'])):
    img = Image.fromarray(data_test['data'][i])
    cls = list(data_test['catname2label'].keys())[list(data_test['catname2label'].values()).index(data_test['labels'][i])]
    img_folder = os.path.join(folder, cls)
    img_name = os.path.join(img_folder, '{:05d}.png'.format(i))
    img.save(img_name)
