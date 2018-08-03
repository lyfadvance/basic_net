import tensorflow as tf
import numpy as np
from mynetwork import Network
import os
import os.path as osp
from imdb_load import imdb_load
from timer import Timer
LEARNING_RATE=0.00001
DISPLAY=5
ROOT_DIR=osp.abspath(osp.join(osp.dirname(__file__)))
DATA_DIR=osp.abspath(osp.join(ROOT_DIR))
class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 2], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
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
        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info')
             .DatalabelToTrainlabel_layer(name = 'rpn-data'))


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
             .spatial_softmax(name='rpn_cls_prob2'))
###################################################
#计算loss
###################################################
    def build_loss(self):
        conloss=tf.reshape(self.get_output('conloss'),[-1])
        conloss=tf.reduce_mean(tf.square(conloss))
        regconloss=tf.reshape(self.get_output('regconloss'),[-1])
        regconloss=tf.reduce_mean(tf.square(regconloss))
        
        conloss2=tf.reshape(self.get_output('conloss2'),[-1])
        conloss2=tf.reduce_mean(tf.square(conloss2))
        regconloss2=tf.reshape(self.get_output('regconloss2'),[-1])
        regconloss2=tf.reduce_mean(tf.square(regconloss2))

        rpn_cls_score=tf.reshape(self.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_cls_score2=tf.reshape(self.get_output('rpn_cls_score_reshape2'),[-1,2])
        rpn_edge_score=tf.reshape(self.get_output('rpn_edge_score_reshape'),[-1,2])
        rpn_label=tf.reshape(self.get_output('rpn-data')[0],[-1])
        rpn_edge_label=tf.reshape(self.get_output('rpn-data')[1],[-1])
        ##收集不是dont care的label
        rpn_keep=tf.where(tf.not_equal(rpn_label,-1))
        ##收集正样本
        fg_keep=tf.equal(rpn_label,1)

        rpn_cls_score=tf.gather(rpn_cls_score,rpn_keep)
        rpn_cls_score2=tf.gather(rpn_cls_score2,rpn_keep)
        rpn_edge_score=tf.gather(rpn_edge_score,rpn_keep)

        rpn_label=tf.gather(rpn_label,rpn_keep)
        rpn_edge_label=tf.gather(rpn_edge_label,rpn_keep)

        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label,logits=rpn_cls_score)
        #计算score loss
        rpn_cross_entropy=tf.reduce_mean(rpn_cross_entropy_n)
        rpn_cross_entropy_n2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label,logits=rpn_cls_score2)
        rpn_cross_entropy2=tf.reduce_mean(rpn_cross_entropy_n2)
        
        rpn_cross_entropy_n_edge = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_edge_label,logits=rpn_edge_score)
        #计算score loss
        rpn_cross_entropy_edge=tf.reduce_mean(rpn_cross_entropy_n_edge)
        #计算box loss
        rpn_bbox_targets=self.get_output('rpn-data')[2]
        rpn_bbox_inside_weights = self.get_output('rpn-data')[3]
        rpn_bbox_outside_weights = self.get_output('rpn-data')[4]

        rpn_bbox_pred=self.get_output('rpn_bbox_pred')
        rpn_bbox_pred2=self.get_output('rpn_bbox_pred2')

        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep) # shape (N, 4)
        rpn_bbox_pred2 = tf.gather(tf.reshape(rpn_bbox_pred2, [-1, 4]), rpn_keep) # shape (N, 4)
        rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
        rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
        rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)
        #print('_______________________________________get')
        rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * self.smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])
        #print('_______________________________________get2')
        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
        rpn_loss_box_n2 = tf.reduce_sum(rpn_bbox_outside_weights * self.smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred2 - rpn_bbox_targets)), reduction_indices=[1])
        rpn_loss_box2 = tf.reduce_sum(rpn_loss_box_n2) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
        #计算score和box的总的loss
        model_loss=rpn_cross_entropy+rpn_loss_box+5*conloss+5*regconloss+rpn_cross_entropy2+rpn_loss_box2+rpn_cross_entropy_edge
        regularization_losses=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss=tf.add_n(regularization_losses)+model_loss

        return total_loss,model_loss,rpn_cross_entropy,rpn_cross_entropy_edge,rpn_loss_box,conloss,regconloss,rpn_cross_entropy2,rpn_loss_box2

#########################################################
#训练网络
#########################################################
    def snapshot(self,sess,iter):
        #print('_______________________get')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        filename=('iter_{:d}'.format(iter+1)+'.ckpt')
        filename=os.path.join(self.output_dir,filename)
        self.saver.save(sess,filename)
        print('Wrote snapshot to :{:s}'.format(filename))

    def train(self,imdb,output_dir,log_dir,pretrained_model=None,max_iters=40000,restore=False):
        config=tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type='BFC'
        config.gpu_options.per_process_gpu_memory_fraction=0.75
        
        with tf.Session(config=config) as sess:
            #####################
            #保存模型和训练数据
            ###############
            self.output_dir=output_dir
            self.log_dir=log_dir
            self.pretrained_model=pretrained_model

            #self.writer必须在创建会话后创建
            self.saver = tf.train.Saver(max_to_keep=10,write_version=tf.train.SaverDef.V2)
            self.writer = tf.summary.FileWriter(logdir=log_dir,
                                                 graph=tf.get_default_graph(),
                                                 flush_secs=5)
            #开始会话
            total_loss,model_loss,rpn_cross_entropy,rpn_cross_entropy_edge,rpn_loss_box,conloss,regconloss,rpn_cross_entropy2,rpn_loss_box2=self.build_loss()
            tf.summary.scalar('rpn_cls_loss',rpn_cross_entropy)
            tf.summary.scalar('model_loss',model_loss)
            tf.summary.scalar('total_loss',total_loss)
            tf.summary.scalar('rpn_loss_box',rpn_loss_box)
            tf.summary.scalar('conloss',conloss)
            tf.summary.scalar('regconloss',regconloss)
            tf.summary.scalar('rpn_edge_loss',rpn_cross_entropy_edge)
            tf.summary.scalar('rpn_cls_loss2',rpn_cross_entropy2)
            tf.summary.scalar('rpn_loss_box2',rpn_loss_box2)
            summary_op=tf.summary.merge_all()

            lr=tf.Variable(LEARNING_RATE,trainable=False)
            opt=tf.train.AdamOptimizer(LEARNING_RATE)
            
            global_step=tf.Variable(0,trainable=False)
            train_op=opt.minimize(total_loss,global_step=global_step)

            sess.run(tf.global_variables_initializer())
            restore_iter=0
            #load VGG16
            if self.pretrained_model is not None and not restore:
                try:
                    print(('loading pretrained model''weights from {:s}').format(self.pretrained_model))
                    self.load(self.pretrained_model,sess,True)
                except:
                    raise 'Check your pretrained model {:s}'.format(self.pretrained_model)
            if restore:
                try:
                    #print(self.output_dir)
                    ckpt=tf.train.get_checkpoint_state(self.output_dir)
                    print('Restoring from {}...'.format(ckpt.model_checkpoint_path),end=' ')
                    self.saver.restore(sess,ckpt.model_checkpoint_path)
                    #获取不带后缀的文件名
                    stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                    restore_iter=int(stem.split('_')[-1])
                    sess.run(global_step.assign(restore_iter))
                    print('done')
                except:
                    raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
            timer=Timer()
            for iter in range(restore_iter,max_iters):
                timer.tic()
                #源代码中学习率调整

                blobs=imdb.forward()
                #print(blobs)
                feed_dict={
                    self.data:blobs['data'],
                    self.im_info:blobs['im_info'],
                    self.gt_boxes:blobs['gt_boxes'],
                }
                #total_loss,model_loss,rpn_cross_entropy,rpn_cross_entropy_edge,rpn_loss_box,conloss,regconloss,rpn_cross_entropy2,rpn_loss_box2
                fetch_list=[total_loss,model_loss,rpn_cross_entropy,rpn_cross_entropy_edge,rpn_loss_box,conloss,regconloss,rpn_cross_entropy2,rpn_loss_box2,summary_op,train_op]
                #print('________________________________start')
                total_loss_val,model_loss_val,rpn_loss_cls_val,_,rpn_loss_box_val,conloss_val,regconloss_val,rpn_loss_cls_val2,rpn_loss_box_val2,summary_str,_=sess.run(fetches=fetch_list,feed_dict=feed_dict)
                self.writer.add_summary(summary=summary_str,global_step=global_step.eval())
                #print('_______________________________-get')
                _diff_time=timer.toc(average=False)

                if (iter) % (DISPLAY) == 0:
                    print('iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f,conloss: %.4f,regconloss:%.4f,rpn_loss_cls2: %.4f,rpn_loss_box2: %.4f,lr: %f' %\
                        (iter, max_iters, total_loss_val,model_loss_val,rpn_loss_cls_val,rpn_loss_box_val,conloss_val,regconloss_val,rpn_loss_cls_val2,rpn_loss_box_val2,lr.eval()))
                    print('speed: {:.3f}s / iter'.format(_diff_time))
                if (iter+1) %500 ==0:
                    last_snap_shot_iter=iter
                    self.snapshot(sess,iter)

if __name__=='__main__':
    net=VGGnet_train()
    imdb=imdb_load()
    print("数据库载入成功")
    output_dir=os.path.join(DATA_DIR,'snapshot')
    print("_______________________",output_dir)
    log_dir=os.path.join(DATA_DIR,'log')
    pretrained_model='VGG_imagenet.npy'
    net.train(imdb,output_dir,log_dir,pretrained_model,restore=False)
