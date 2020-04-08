##### Deep dive into SSD training: 3 tips to boost performance


### Loss normalization: use batch-wise norm instead of sample-wise norm

import mxnet as mx
x = mx.random.uniform(shape=(2, 3, 300, 300))  # use batch-size 2
x.shape

# suppose image 1 has single object
id1 = mx.nd.array([1])
bbox1 = mx.nd.array([[10, 20, 80, 90]])         # xmin, ymin, xmax, ymax

# suppose image 2 has 4 objects
id2 = mx.nd.array([1, 3, 5, 7])
bbox2 = mx.nd.array([[10, 10, 30, 30], [40, 40, 60, 60], [50, 50, 90, 90], [100, 110, 120, 140]])

# combine them into a batch by padding -1 as sentinal values
gt_ids = mx.nd.ones(shape=(2, 4)) * -1
gt_ids[0, :1] = id1
gt_ids[1, :4] = id2
print('class_ids:', gt_ids)

gt_boxes = mx.nd.ones(shape=(2, 4, 4)) * -1
gt_boxes[0, :1, :] = bbox1
gt_boxes[1, :, :] = bbox2
print('bounding boxes:', gt_boxes)

# vgg16모델을 사용하고 300x300 SSD model
from gluoncv import model_zoo
net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False, pretrained=False)

from mxnet import gluon
net.initialize()
conf_loss = gluon.loss.SoftmaxCrossEntropyLoss()
loc_loss = gluon.loss.HuberLoss()

# Simulate the training steps by manually compute losses:
# You can always use gluoncv.loss.SSDMultiBoxLoss which fulfills this function.
from mxnet import autograd
from gluoncv.model_zoo.ssd.target import SSDTargetGenerator
target_generator = SSDTargetGenerator()

with autograd.record():
    # 1. forward pass
    cls_preds, box_preds, anchors = net(x)

    # 2. generate training targets
    cls_targets, box_targets, box_masks = target_generator(anchors, cls_preds, gt_boxes, gt_ids)
    num_positive = (cls_targets > 0).sum().asscalar()
    cls_mask = (cls_targets >= 0).expand_dims(axis=-1)  # negative targets should be ignored in loss

    # 3 losses, here we have two options, batch-wise or sample wise norm
    # 3.1 batch wise normalization: divide loss by the summation of num positive targets in batch
    batch_conf_loss = conf_loss(cls_preds, cls_targets, cls_mask) / num_positive
    batch_loc_loss = loc_loss(box_preds, box_targets, box_masks) / num_positive

    # 3.2 sample wise normalization: divide by num positive targets in this sample(image)
    sample_num_positive = (cls_targets > 0).sum(axis=0, exclude=True)
    sample_conf_loss = conf_loss(cls_preds, cls_targets, cls_mask) / sample_num_positive
    sample_loc_loss = loc_loss(box_preds, box_targets, box_masks) / sample_num_positive

    # Since ``conf_loss`` and ``loc_loss`` calculate the mean of such loss, we want
    # to rescale it back to loss per image.
    rescale_conf = cls_preds.size / cls_preds.shape[0]
    rescale_loc = box_preds.size / box_preds.shape[0]
    # then call backward and step, to update the weights, etc..
    # L = conf_loss + loc_loss * alpha
    # L.backward()



# The norms are different, but sample-wise norms sum up to be the same with batch-wise norm
print('batch-wise num_positive:', num_positive)
print('sample-wise num_positive:', sample_num_positive)

# Compare the losses
print('batch-wise norm conf loss:', batch_conf_loss * rescale_conf)
print('sample-wise norm conf loss:', sample_conf_loss * rescale_conf)

print('batch-wise norm loc loss:', batch_loc_loss * rescale_loc)
print('sample-wise norm loc loss:', sample_loc_loss * rescale_loc)


### Initializer matters: don’t stick to one single initializer
from gluoncv import model_zoo
import mxnet as mx
# don't load pretrained for this demo
net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained=False, pretrained_base=False)
# random init
net.initialize()
# gluon only infer shape when real input data is used
net(mx.nd.zeros(shape=(1, 3, 300, 300)))
# now we have real shape for each parameter
predictors = [(k, v) for k, v in net.collect_params().items() if 'predictor' in k]
name, pred = predictors[0]
print(name, pred)


# we can initialize it with different initializers, such as Normal or Xavier.
pred.initialize(mx.init.Uniform(), force_reinit=True)
print('param shape:', pred.data().shape, 'peek first 20 elem:', pred.data().reshape((-1))[:20])

# Simply switching from Uniform to Xavier can produce ~1% mAP gain.
pred.initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2, factor_type='out'), force_reinit=True)
print('param shape:', pred.data().shape, 'peek first 20 elem:', pred.data().reshape((-1))[:20])



### Interpreting confidence scores: process each class separately
print('class prediction shape:', cls_preds.shape)

cls_pred = mx.nd.array([-1, -2, 3, 4, 6.5, 6.4])
cls_prob = mx.nd.softmax(cls_pred, axis=-1)
for k, v in zip(['bg', 'apple', 'orange', 'person', 'dog', 'cat'], cls_prob.asnumpy().tolist()):
    print(k, v)

