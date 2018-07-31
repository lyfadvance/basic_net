import tensorflow as tf
import numpy as np
from mynetwork import Network
import os
import os.path as osp
from imdb_load import imdb_load
from timer import Timer
LEARNING_RATE=0.00001
DISPLAY=100
ROOT_DIR=osp.abspath(osp.join(osp.dirname(__file__)))
DATA_DIR=osp.abspath(osp.join(ROOT_DIR))
class RESnet_train(Network):
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
             .conv(7, 7, 64, 2, 2, relu=False, name='conv1')
             .batch_normalization(relu=True, name='bn_conv1', is_training=False)
             .max_pool(3, 3, 2, 2, padding='VALID',name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(name='bn2a_branch1',is_training=False,relu=False))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(relu=True, name='bn2a_branch2a',is_training=False)
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(relu=True, name='bn2a_branch2b',is_training=False)
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(name='bn2a_branch2c',is_training=False,relu=False))

        (self.feed('bn2a_branch1', 
                   'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .batch_normalization(relu=True, name='bn2b_branch2a',is_training=False)
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .batch_normalization(relu=True, name='bn2b_branch2b',is_training=False)
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
             .batch_normalization(name='bn2b_branch2c',is_training=False,relu=False))

        (self.feed('res2a_relu', 
                   'bn2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
             .batch_normalization(relu=True, name='bn2c_branch2a',is_training=False)
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
             .batch_normalization(relu=True, name='bn2c_branch2b',is_training=False)
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
             .batch_normalization(name='bn2c_branch2c',is_training=False,relu=False))

        (self.feed('res2b_relu', 
                   'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1', padding='VALID')
             .batch_normalization(name='bn3a_branch1',is_training=False,relu=False))

        (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a', padding='VALID')
             .batch_normalization(relu=True, name='bn3a_branch2a',is_training=False)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(relu=True, name='bn3a_branch2b',is_training=False)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
             .batch_normalization(name='bn3a_branch2c',is_training=False,relu=False))

        (self.feed('bn3a_branch1', 
                   'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
             .batch_normalization(relu=True, name='bn3b_branch2a',is_training=False)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
             .batch_normalization(relu=True, name='bn3b_branch2b',is_training=False)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
             .batch_normalization(name='bn3b_branch2c',is_training=False,relu=False))

        (self.feed('res3a_relu', 
                   'bn3b_branch2c')
             .add(name='res3b')
             .relu(name='res3b_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
             .batch_normalization(relu=True, name='bn3c_branch2a',is_training=False)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')
             .batch_normalization(relu=True, name='bn3c_branch2b',is_training=False)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
             .batch_normalization(name='bn3c_branch2c',is_training=False,relu=False))

        (self.feed('res3b_relu', 
                   'bn3c_branch2c')
             .add(name='res3c')
             .relu(name='res3c_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
             .batch_normalization(relu=True, name='bn3d_branch2a',is_training=False)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')
             .batch_normalization(relu=True, name='bn3d_branch2b',is_training=False)
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
             .batch_normalization(name='bn3d_branch2c',is_training=False,relu=False))

        (self.feed('res3c_relu', 
                   'bn3d_branch2c')
             .add(name='res3d')
             .relu(name='res3d_relu')
             .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1', padding='VALID')
             .batch_normalization(name='bn4a_branch1',is_training=False,relu=False))

        (self.feed('res3d_relu')
             .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a', padding='VALID')
             .batch_normalization(relu=True, name='bn4a_branch2a',is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(relu=True, name='bn4a_branch2b',is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
             .batch_normalization(name='bn4a_branch2c',is_training=False,relu=False))

        (self.feed('bn4a_branch1', 
                   'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
             .batch_normalization(relu=True, name='bn4b_branch2a',is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
             .batch_normalization(relu=True, name='bn4b_branch2b',is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c')
             .batch_normalization(name='bn4b_branch2c',is_training=False,relu=False))

        (self.feed('res4a_relu', 
                   'bn4b_branch2c')
             .add(name='res4b')
             .relu(name='res4b_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
             .batch_normalization(relu=True, name='bn4c_branch2a',is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4c_branch2b')
             .batch_normalization(relu=True, name='bn4c_branch2b',is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c')
             .batch_normalization(name='bn4c_branch2c',is_training=False,relu=False))

        (self.feed('res4b_relu', 
                   'bn4c_branch2c')
             .add(name='res4c')
             .relu(name='res4c_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a')
             .batch_normalization(relu=True, name='bn4d_branch2a',is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4d_branch2b')
             .batch_normalization(relu=True, name='bn4d_branch2b',is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c')
             .batch_normalization(name='bn4d_branch2c',is_training=False,relu=False))

        (self.feed('res4c_relu', 
                   'bn4d_branch2c')
             .add(name='res4d')
             .relu(name='res4d_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a')
             .batch_normalization(relu=True, name='bn4e_branch2a',is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4e_branch2b')
             .batch_normalization(relu=True, name='bn4e_branch2b',is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c')
             .batch_normalization(name='bn4e_branch2c',is_training=False,relu=False))

        (self.feed('res4d_relu', 
                   'bn4e_branch2c')
             .add(name='res4e')
             .relu(name='res4e_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a')
             .batch_normalization(relu=True, name='bn4f_branch2a',is_training=False)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4f_branch2b')
             .batch_normalization(relu=True, name='bn4f_branch2b',is_training=False)
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c')
             .batch_normalization(name='bn4f_branch2c',is_training=False,relu=False))

        (self.feed('res4e_relu', 
                   'bn4f_branch2c')
             .add(name='res4f')
             .relu(name='res4f_relu'))

        

        #========= RPN ============
        (self.feed('res4f_relu')
             .conv(3,3,512,1,1,name='rpn_conv/3x3'))
             #.conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))

        #(self.feed('rpn_conv/3x3').Bilstm(512,128,512,name='lstm_o'))
        (self.feed('rpn_conv/3x3').regress(512,1 * 2, name='rpn_cls_score')
                                  .conloss_conv(3,3,2,3,3,name='conloss'))
        (self.feed('rpn_conv/3x3').regress(512,1 * 4, name='rpn_bbox_pred')
                                  .regconloss_conv(3,3,2,3,3,name='regconloss'))
        #(self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 2,name='rpn_cls_score'))

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
###################################################
#计算loss
###################################################
    def build_loss(self):
        conloss=tf.reshape(self.get_output('conloss'),[-1])
        conloss=tf.reduce_mean(tf.square(conloss))
        regconloss=tf.reshape(self.get_output('regconloss'),[-1])
        regconloss=tf.reduce_mean(tf.square(regconloss))

        rpn_cls_score=tf.reshape(self.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label=tf.reshape(self.get_output('rpn-data')[0],[-1])
        ##收集不是dont care的label
        rpn_keep=tf.where(tf.not_equal(rpn_label,-1))
        ##收集正样本
        fg_keep=tf.equal(rpn_label,1)

        rpn_cls_score=tf.gather(rpn_cls_score,rpn_keep)
        rpn_label=tf.gather(rpn_label,rpn_keep)
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label,logits=rpn_cls_score)
        #计算score loss
        rpn_cross_entropy=tf.reduce_mean(rpn_cross_entropy_n)
        
        #计算box loss
        rpn_bbox_targets=self.get_output('rpn-data')[1]
        rpn_bbox_inside_weights = self.get_output('rpn-data')[2]
        rpn_bbox_outside_weights = self.get_output('rpn-data')[3]

        rpn_bbox_pred=self.get_output('rpn_bbox_pred')
        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep) # shape (N, 4)
        rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
        rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
        rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)
        rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * self.smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])
        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
        #计算score和box的总的loss
        model_loss=rpn_cross_entropy+rpn_loss_box+5*conloss+5*regconloss
        regularization_losses=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss=tf.add_n(regularization_losses)+model_loss

        return total_loss,model_loss,rpn_cross_entropy,rpn_loss_box,conloss,regconloss

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
            total_loss,model_loss,rpn_cross_entropy,rpn_loss_box,conloss,regconloss=self.build_loss()
            tf.summary.scalar('rpn_cls_loss',rpn_cross_entropy)
            tf.summary.scalar('model_loss',model_loss)
            tf.summary.scalar('total_loss',total_loss)
            tf.summary.scalar('rpn_loss_box',rpn_loss_box)
            tf.summary.scalar('conloss',conloss)
            tf.summary.scalar('regconloss',regconloss)
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

                fetch_list=[total_loss,model_loss,rpn_cross_entropy,summary_op,train_op]
                total_loss_val,model_loss_val,rpn_loss_cls_val,summary_str,_=sess.run(fetches=fetch_list,feed_dict=feed_dict)
                self.writer.add_summary(summary=summary_str,global_step=global_step.eval())
                _diff_time=timer.toc(average=False)

                if (iter) % (DISPLAY) == 0:
                    print('iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, lr: %f'%\
                        (iter, max_iters, total_loss_val,model_loss_val,rpn_loss_cls_val,lr.eval()))
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
    pretrained_model='Resnet50.npy'
    net.train(imdb,output_dir,log_dir,pretrained_model,restore=False)