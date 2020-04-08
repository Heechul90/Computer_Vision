##### 2. Predict with pre-trained Faster RCNN model


### import 필수 라이브러리
from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils


### Load a pretrained model
net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)


### Pre-process an image
im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/biking.jpg?raw=true',
                          path='img/biking.jpg')
# im_fname1 = 'img/biking.jpg'
# im_fname2 = 'img/street_small.jpg'
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname) # 리스트로 들어가도 됨([im_fname1, im_fname2])


### Inference and display
box_ids, scores, bboxes = net(x)
ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

plt.show()