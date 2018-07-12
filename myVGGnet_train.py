import tensorflow as tf
import numpy as np
from .mynetwork import Network

class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

    def setup(self):

        # n_classes = 21
        n_classes = cfg.NCLASSES
        # anchor_scales = [8, 16, 32]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]

        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))
        #========= RPN ============
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='rpn_conv/3x3'))

        #(self.feed('rpn_conv/3x3').Bilstm(512,128,512,name='lstm_o'))
        (self.feed('rpn_conv/3x3').regress(512,10 * 4, name='rpn_cls_score'))
        #(self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 2,name='rpn_cls_score'))

        # generating training labels on the fly
        # output: rpn_labels(HxWxA, 2) rpn_bbox_targets(HxWxA, 4) rpn_bbox_inside_weights rpn_bbox_outside_weights
        # 给每个anchor上标签，并计算真值（也是delta的形式），以及内部权重和外部权重
        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info')
             .DatalabelToTrainlabel_layer(name = 'rpn-data' ))

        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        # 给之前得到的score进行softmax，得到0-1之间的得分
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape')
             )
    def build_loss(self):
        rpn_cls_score=tf.reshape(self.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label=tf.reshape(self.get_output('rpn-data')[0],[-1])
        ##收集不是dont care的label
        rpn_keep=tf.where(tf.not_equal(rpn_label,-1))
        rpn_cls_socre=tf.gather(rpn_cls_socre,rpn_keep)
        rpn_label=tf.gather(rpn_label,rpn_keep)
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label,logits=rpn_cls_score)
        rpn_cross_entropy=tf.reduce_mean(rpn_cross_entropy_n)
        
        model_loss=rpn_cross_entropy
        regularization_losses=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss=tf.add_n(regularization_losses)+model_loss

        return total_loss,model_loss,rpn_cross_entropy
