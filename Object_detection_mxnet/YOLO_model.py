##### 3. Predict with pre-trained YOLO models


### import 필수 라이브러리
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt


### Load a pretrained model
net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)


### Pre-process an image
im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/dog.jpg',
                          path='img/dog.jpg')
x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)


### Inference and display
class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()