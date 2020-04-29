import json
import mxnet as mx
from mxnet import gluon, nd
import numpy as np


########################################################################################################################
########################### 데이터 불러오기: 이미지는 리스트로 불러옵니다.
from cnn.load_image import Preprocessing

path = 'dataset/image/pretrained/'
image_resize = 224

a = Preprocessing()
a.setdata(path, image_resize)
image_list = a.load_image()
print(image_list[0])

########################### Loading the model: trained된 모델 불러옵니다.
ctx = mx.cpu()

new_net = build_lenet(gluon.nn.Sequential())


########################### Loading the categories: 이미저장된 imagenet labels을 불러옵니다.
categories = np.array(json.load(open('image_net_labels.json', 'r')))

########################### defined predict
def predict(model, image, categories, k):
    predictions = model(image).softmax()
    top_pred = predictions.topk(k=k)[0].asnumpy()
    for index in top_pred:
        probability = predictions[0][int(index)]
        category = categories[int(index)]
        print("{}: {:.2f}%".format(category, probability.asscalar()*100))
    print('')

########################### predict(모델, 이미지 리스트, 카테고리, 평가항목 순위3까지)
for image in image_list:
    predict(resnet, image, categories, 3)
