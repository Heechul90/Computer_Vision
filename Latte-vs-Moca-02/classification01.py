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
# from mxnet.gluon.model_zoo import vision
# ctx = mx.cpu()
# net = vision.resnet18_v1(classes=1, pretrained=False)

net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2), nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(1))




################## accuracy
# def evaluate_accuracy(data_iterator, net, ctx):
#     acc = mx.metric.Accuracy()
#     for i, (data, label) in enumerate(data_iterator):
#         predictions = nd.argmax(net(data.as_in_context(ctx)), axis=1)
#         acc.update(preds=predictions, labels=label.as_in_context(ctx))
#     return acc.get()[1]

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis = 1)
        acc.update(preds = predictions, labels = label)
    return acc.get()


################## parameters
mx.random.seed(1)
epochs = 10
lr = 0.01
num_workers = 0
loss_function = gluon.loss.SigmoidBinaryCrossEntropyLoss()
metric = mx.metric.Accuracy()

# Initialize parameters randomly
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

#############################################################################################
# train_start = time()
# all_train_mse = []
# all_test_mse = []
# train_imgs = []
#
# for e in range(epochs):
#     for i, (data, label) in enumerate(train_iter):
#         data = data.as_in_context(ctx)
#         label = label.as_in_context(ctx)
#         # Wait for completion of previous iteration to
#         # avoid unnecessary memory allocation
#         nd.waitall()
#         with autograd.record():
#             output = net(data)
#             loss = loss_function(output, label)
#         loss.backward()
#         trainer.step(data.shape[0])
#         # metric.update([label], [output])
#         # if i % 10 == 0 and i > 0:
#         #     name, acc = metric.get()
#         #     print('[Epoch %d Batch %d] Training: %s=%f' % (e, i, name, acc))
#
#     train_mse = evaluate_accuracy(train_iter, net, ctx)
#     test_mse = evaluate_accuracy(test_iter, net, ctx)
#     all_train_mse.append(train_mse)
#     all_test_mse.append(test_mse)
#
#     name, acc = metric.get()
#     print('[Epoch %d] Training: %s=%f' % (e, name, acc))


epochs = 10
smoothing_constant = 0.01

for e in range(epochs):
    for i, (d, l) in enumerate(train_iter):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = loss_function(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        ############
        # keep a moving average of the losses
        ############

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(test_iter, net)
    train_accuracy = evaluate_accuracy(train_iter, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

file_name = 'net.params'
net.save_parameters(file_name)

net.load_parameters(file_name)

## Predict
def transform(image):
    resized = mx.image.resize_short(image, image_resize)  # 최소 224x224 사이즈이어야 함(inceptionv3는 299 사이즈)
    cropped, crop_info = mx.image.center_crop(resized, (224, 224))
    normalized = mx.image.color_normalize(cropped.astype(np.float32) / 255,
                                          mean=mx.nd.array([0.485, 0.456, 0.406]),
                                          std=mx.nd.array([0.229, 0.224, 0.225]))
    # 4원 형태로 변환(N,3,224,224)
    transposed = normalized.transpose((2, 0, 1))  # Transposing from (224, 224, 3) to (3, 224, 224)
    batchified = transposed.expand_dims(axis=0)  # change the shape from (3, 224, 224) to (1, 3, 224, 224)
    return batchified


image = mx.image.imread('Latte-vs-Moca/dataset/Latte/KakaoTalk_20200414_115510045_01.jpg')
image = mx.image.imread('Latte-vs-Moca/dataset/Moka/KakaoTalk_20200414_120607989_20.jpg')
image = transform(image)
image

pred = net(image).argmax(axis=1)
pred = pred.astype('int32').asscalar()
from IPython import display

_, figs = plt.subplots(1, 10, figsize=(15, 15))
categories = ['Latte', 'Moca']
display.set_matplotlib_formats('svg')


figs.imshow(image.reshape((28,28)).asnumpy())
ax = f.axes
ax.set_title(text_labels[yi]+'\n'+text_labels[pyi])
ax.title.set_fontsize(14)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

categories = ['Latte', 'Moca']






