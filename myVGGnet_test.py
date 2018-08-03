import tensorflow as tf
import numpy as np
from mynetwork import Network
import glob
import os
import os.path as osp
from imdb_load import imdb_load
from timer import Timer
from PIL import Image
from util.draw_box import Drawbox
import cv2
LEARNING_RATE=0.00001
DISPLAY=1
ROOT_DIR=osp.abspath(osp.join(osp.dirname(__file__)))
DATA_DIR=osp.abspath(osp.join(ROOT_DIR))
class VGGnet_test(Network):
    def __init__(self, trainable=False):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 2], name='im_info')
        self.layers = dict({'data':self.data, 'im_info':self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):

        # n_classes = 21
        n_classes = 2
        # anchor_scales = [8, 16, 32]
        #anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]

        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .conv(3,3,3,1,1,name='convm_1')
             )
        (self.feed('data','convm_1')
             .mask(name='mask')
             .reuse_conv(3,3,64,1,1,'conv1_1',name='conv01_1')
             .reuse_conv(3,3,64,1,1,'conv1_2',name='conv01_2')
             
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
        (self.feed('rpn_conv/3x3').regress(512,1 * 2, name='rpn_cls_score')
                                  .conloss_conv(3,3,2,2,2,name='conloss'))
        (self.feed('rpn_conv/3x3').regress(512,1*2,name='rpn_edge_score'))
        (self.feed('rpn_conv/3x3').regress(512,1 * 4, name='rpn_bbox_pred')
                                  .regconloss_conv(3,3,2,2,2,name='regconloss'))
        #(self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 2,name='rpn_cls_score'))
        (self.feed('rpn_cls_score','rpn_edge_score','rpn_bbox_pred')
             .concat(axis=3,name='myconcat'))
        (self.feed('myconcat')
             .conv(3,3,1*2,1,1,relu=False,name='rpn_cls_score2')
             .conloss_conv(3,3,2,2,2,name='conloss2')
             )
        (self.feed('myconcat')
            .conv(3,3,1*4,1,1,relu=False,name='rpn_bbox_pred2')
            .regconloss_conv(3,3,2,2,2,name='regconloss2'))

        # generating training labels on the fly
        # output: rpn_labels(HxWxA, 2) rpn_bbox_targets(HxWxA, 4) rpn_bbox_inside_weights rpn_bbox_outside_weights
        # 给每个anchor上标签，并计算真值（也是delta的形式），以及内部权重和外部权重
        #(self.feed('rpn_cls_score', 'gt_boxes', 'im_info')
         #    .DatalabelToTrainlabel_layer(name = 'rpn-data'))


        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        # 给之前得到的score进行softmax，得到0-1之间的得分
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape')#[1,H,W*A,4]
             .spatial_softmax(name='rpn_cls_prob'))
        (self.feed('rpn_edge_score')
             .spatial_reshape_layer(2,name='rpn_edge_score_reshape')
             .spatial_softmax(name='rpn_edge_prob'))

        (self.feed('rpn_cls_score2')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape2')#[1,H,W*A,4]
             .spatial_softmax(name='rpn_edge_prob2'))

        (self.feed('rpn_edge_prob2')
            .spatial_reshape_layer(1*2,name='rpn_cls_prob_reshape'))
        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred2','im_info')
            .TrainlabelToDatalabel_layer(name='rois'))

#################################################
#载入网络
#################################################
    def load_model(self):
        config=tf.ConfigProto(allow_soft_placement=True)
        sess=tf.Session(config=config)
        
        saver=tf.train.Saver()
        try:
            checkpoint_path=os.path.join(DATA_DIR,'snapshot')
            ckpt=tf.train.get_checkpoint_state(checkpoint_path)
            print('Restoring from{}...'.format(ckpt.model_checkpoint_path),end=' ')
            saver.restore(sess,ckpt.model_checkpoint_path)
            print('done')
        except:
            raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
        return sess
    
    ##载入数据
    def load_image(self):
        im_names=glob.glob(os.path.join(DATA_DIR,'test','*.jpg'))
        return im_names
    ##评价
    def evaluation(self):
        sess=self.load_model()
        im_names=self.load_image()
        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(('Demo for {:s}'.format(im_name)))
            result=self.test_single(sess,im_name)
            rois=result[0]
            
            tps=rois[:,0:4]
            scores=rois[:,4]
            Draw=Drawbox()
            boxes=Draw.draw_boxes(im_name,tps,1,scores,draw=False)
            basename=os.path.basename(im_name)
            with open('results/'+'res'+'_'+basename+'.txt','w') as f:
                for j in range(len(boxes)):
                    f.write(str(boxes[j][0]))
                    f.write(',')
                    f.write(str(boxes[j][1]))
                    f.write(',')
                    f.write(str(boxes[j][2]))
                    f.write(',')
                    f.write(str(boxes[j][3]))
                    f.write('\n')
    ##测试
    def test(self):
        sess=self.load_model()
        im_names=self.load_image()
        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(('Demo for {:s}'.format(im_name)))
            rois,feature_map=self.test_single(sess,im_name)
            self.show_feature_map(feature_map,im_name,'convm_1')
            #print(rois)
            a=[0,1,2,3]
            tps=rois[:]
            tps=tps[:,a]
            scores=rois[:]
            scores=scores[:,4]
            Draw=Drawbox()
            Draw.draw_boxes(im_name,tps,1,scores)
    ##测试单个图片
    def test_single(self,sess,im_name):
        img=cv2.imread(im_name)
        
        blobs={'data':None,'rois':None}
        blobs['data']=[img]
        blobs['im_info']=np.array([[img.shape[0],img.shape[1]]],dtype=np.float32)

        feed_dict={self.data:blobs['data'],self.im_info:blobs['im_info']}
        
        rois,convm_1=sess.run([net.get_output('rois'),net.get_output('convm_1')],feed_dict=feed_dict)
        #rois=rois[0]
        return rois,convm_1[0]
    def show_feature_map(self,feature_map,im_name,feature_name):
        feature_map=feature_map*255
        height,width,depth=feature_map.shape
        print(depth)
        for i in range(depth):
            image=Image.fromarray(feature_map[:,:,i].astype(np.uint8))
            basename=os.path.basename(im_name)
            basename=basename.split('.')
            if not os.path.exists(os.path.join('results',basename[0])):
                os.makedirs(os.path.join('results',basename[0]))
            image.save(os.path.join('results',basename[0],feature_name+'_'+str(i)+'.jpg'))
    def show_scores(self,scores,im_name,height,width):
        scores=scores*255
        print(height,width)
        image=Image.fromarray(scores.astype(np.uint8))
        image=image.resize((width,height))
        basename=os.path.basename(im_name)
        basename=basename.split('.')
        if  not os.path.exists('results'):
            os.makedirs('results')
        image.save(os.path.join('results',basename[0]+'_result.jpg'))
            
if __name__=='__main__':
    net=VGGnet_test()
    print("模型构建完成")
    net.test()

