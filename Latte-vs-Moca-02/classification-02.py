### 라떼와 모카 분류 - 02

# 함수, 모듈 준비
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gluoncv
import mxnet as mx
import numpy as np

from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
from time import time
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from gluoncv.data.transforms.presets.imagenet import transform_eval
from mxnet.gluon.data.vision import datasets, transforms


# 변수
data_path = 'Latte-vs-Moca-02/dataset'
image_resize = 96
ctx = mx.cpu()

# 이미지 데이터 전처리
def transformer(data, label):
    data = mx.image.imresize(data, image_resize, image_resize)
    data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
    label = np.float32(label)
    return data, label

# ImageFolderDataset으로 폴더에 있는 이미지를 불러옴
input_data = gluon.data.vision.datasets.ImageFolderDataset(data_path, transform=transformer)

# 이미지 데이터와 레이블 데이터를 따로 리스트로 저장
img_data = []
img_label = []
for d, l in input_data:
    img_data.append(d)
    img_label.append(l)

# 이미지와 레이블 데이터를 trainset과 testset으로 구분
train_data, test_data, train_label, test_label = train_test_split(img_data, img_label,
                                                                  test_size=0.3,
                                                                  shuffle=True)


# DataLoader함수로 batch로 데이터를 불러옴
batch_size = 32
train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label),
                                   batch_size=batch_size,
                                   shuffle=False)
test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label),
                                  batch_size=batch_size,
                                  shuffle=False)

for d, l in train_iter:
    break


########################################################################################################################
################## model
from mxnet.gluon.model_zoo import vision
ctx = mx.cpu()
net = vision.resnet18_v1(classes=1, pretrained=False)

# net = nn.Sequential()
# with net.name_scope():
#     net.add(
#         nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
#         nn.MaxPool2D(pool_size=2, strides=2),
#         nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
#         nn.MaxPool2D(pool_size=2, strides=2), nn.Flatten(),
#         nn.Dense(128, activation="relu"),
#         nn.Dense(1))




################## accuracy







