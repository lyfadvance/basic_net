import numpy as np
import numpy.random as npr

def TrainlabelToDatalabel_layer(rpn_cls_prob_reshape,rpn_bbox_pred,im_info):
    _anchors=np.array([[0,0,15,15]],np.int32)
    _num_anchors=_anchors.shape[0]
    height,width=rpn_cls_prob_reshape.shape[1:3]
    K=height*width
    A=_num_anchors


    #print("______________________",height,width)
    scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:,:,:,:,1],
                        [1, height, width, _num_anchors])
    scores=np.reshape(scores,[height,width])

####################################################
    #测试ctpn构造的anchor方法,然后实现自己的方法
####################################################
    shift_x = np.arange(0, width) * 16
    shift_y = np.arange(0, height) * 16
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) # in W H order
    # K is H x W
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()#生成feature-map和真实image上anchor之间的偏移量
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))#相当于复制宽高的维度，然后相加
    all_anchors = all_anchors.reshape((K * A, 4))
#######################################################
    #print(all_anchors)
    bbox_deltas=rpn_bbox_pred
    bbox_deltas=bbox_deltas.reshape((-1,4))
    scores=scores.reshape((-1,1))
    proposals=bbox_transform_inv(anchors,bbox_deltas)
    proposals=clip_boxes(proposals,im_info[:2])#将所有的proposal修建一下，超出图像范围的将会被修剪掉
    #ctpn代码中还做了一些过滤,这里没有实现
    order=scores.ravel().argsort()[::-1]
    order=order[:30*30*1]#这里有bug,因为现在没有实现图像缩放，生成的score个数可能小于30*30*1
    proposals=proposals[order,:]
    scores=scores[order]
    bbox_deltas=bbox_deltas=[order,:]

    ##开始进行nms算法
    ##其中scores形如
#[[],len为1
# []
#  ...
#]
    ##proposals形如:
#[[],len为4
# []
# ...
#]
    #构造nms的输入
    nms_input=np.hstack((proposals,socres))
    blob = np.hstack((scores.astype(np.float32, copy=False), proposals.astype(np.float32, copy=False)))
    return blob,bbox_deltas
#########################
#    根据网络输出，计算box坐标
#########################
def bbox_transform_inv(boxes, deltas):#boxes为10个anchor,deltas为边框回归预测

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes