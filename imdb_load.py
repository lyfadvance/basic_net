#encoding utf-8
import os
import os.path as osp
import numpy as np
import scipy.sparse
import cv2
import PIL.Image
import copy
try:
    import cPickle as pickle
except:
    import pickle

import xml.etree.ElementTree as ET
from imdb import imdb
import matplotlib.pyplot as plt
ROOT_DIR=osp.abspath(osp.join(osp.dirname(__file__)))
DATA_DIR=osp.abspath(osp.join(ROOT_DIR))
BATCH_SIZE=10
class imdb_load(imdb):
    def __init__(self):
        imdb.__init__(self,'dataset')
        self._classes=('background','text')
        self._data_path=self._get_data_path()
        self._image_ext='.jpg'
        self._image_set='train'
        self._image_index=self._load_image_set_index()
        self._roidb_handler=self.gt_roidb
        self.get_training_roidb()
        ###打乱roidb的顺序
        self._shuffle_roidb_inds()
    def image_path_at(self,i):
        return self.image_path_from_index(self._image_index[i])
    def image_path_from_index(self,index):
        image_path=os.path.join(self._data_path,index+self._image_ext)
        assert os.path.exists(image_path),\
                'Path does not exist:{}'.format(image_path)
        return image_path
    #找到数据集的地址
    def _get_data_path(self):
        return os.path.join(DATA_DIR,'train_images')
    def _load_image_set_index(self):
        root=os.path.join(ROOT_DIR,"train_images")
        image_index=[]
        for dirpath,dirnames,filenames in os.walk(root):
            for name in filenames:
                image_index.append(name.split('.')[0])
        image_index=sorted(image_index,key=lambda x:int(x.split('_')[1]))#图片文件名格式形如img_number.jpg,存储的index为img_number
        return image_index
##准备roidb
    def gt_roidb(self):
        gt_roidb=[self._load_pascal_annotation(index) for index in self.image_index]
        return gt_roidb
    def _load_pascal_annotation(self,index):
        filename=os.path.join(DATA_DIR,'annotations',index+'.txt')#annotations形如img_number.jpg
        f = open(filename,'r')
        lines = f.readlines()
        boxes=np.zeros((len(lines),4),dtype=np.uint16)
        for ix,line in enumerate(lines):
            line=line.encode('utf-8').decode('utf-8-sig')
            infos=line.split(',')
            #line=line.encode('utf-8').decode('utf-8-sig')
            ##水平的
            x1 = int(infos[0])
            y1 = int(infos[1])
            x2 = int(infos[2])
            y2 = int(infos[3])
            '''
            x3 = int(infos[4])
            y3 = int(infos[5])
            x4 = int(infos[6])
            y4 = int(infos[7])
            boxes[ix,:]=[x1,y1,x2,y2,x3,y3,x4,y4]
            '''
            boxes[ix,:]=[x1,y1,x2,y2]
        return {'boxes':boxes}
#准备roidb
    def get_training_roidb(self):
        sizes=[PIL.Image.open(self.image_path_at(i)).size
            for i in range(self.num_images)]
        roidb=self.roidb
        for i in range(len(self.image_index)):
            roidb[i]['image']=self.image_path_at(i)
            roidb[i]['width']=sizes[i][0]
            roidb[i]['height']=sizes[i][1]
    ##这里集成batch的获得
    def _shuffle_roidb_inds(self):
        self._perm=np.random.permutation(np.arange(len(self._roidb)))
        self._cur=0
    def _get_next_minibatch_inds(self):
        if self._cur+BATCH_SIZE>= len(self._roidb):
            self._shuffle_roidb_inds()
        db_inds=self._perm[self._cur:self._cur+BATCH_SIZE]
        self._cur+=BATCH_SIZE
        return db_inds
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs
    def _get_next_minibatch(self):
        db_inds=self._get_next_minibatch_inds()
        minibatch_db=[self._roidb[i] for i in db_inds]
        return self.get_minibatch(minibatch_db,self._num_classes)
    def get_minibatch(self,roidb,num_classes):
        num_images=len(roidb)
        im_blob,boxes_scale=self._get_image_blob(roidb)
        blobs={'data':im_blob}
        #print("载入了图像")
        
#assert len(roidb)==1,"Single batch only"
        #这里where的用法颇为奇怪,np.where返回的是(坐标,)所以需要[0]
        #返回的是类别是文本的box
#为兼容多类别的冗余代码
        #gt_inds=np.where(roidb[0]['gt_classes']!=0)[0]
        #gt_boxes=[]
        im_infos=[]
        im_names=[]
        for i in range(num_images):
            #gt_boxes=np.empty((len(roidb[0]['boxes']),8),dtype=np.float32)
            #gt_boxes[:,0:7]=roidb[0]['boxes'][gt_inds,:]
            #gt_boxes[:,0:4]=roidb[0]['boxes']
            #gt_boxes.append(roidb[i]['boxes'])
            im_infos.append([im_blob.shape[1],im_blob.shape[2]])
            im_names.append(os.path.basename(roidb[i]['image']))
       # gt_boxes[:,7]=roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes']=boxes_scale#为list
        '''
        blobs['im_info']=np.array(
            [[im_blob.shape[1],im_blob.shape[2]]],
            dtype=np.float32)
        blobs['im_name']=os.path.basename(roidb[0]['image'])
        '''
        blobs['im_info']=im_infos
        blobs['im_name']=im_names
        #现在除了data是np,其他的都是list
        return blobs
    def _get_image_blob(self,roidb):
        num_images=len(roidb)#这里其实就是等于1,为方便与ctpn的代码结合
        processed_ims=[]
        max_height=0
        max_width=0
        gt_boxes_reshape=[]
        ims=[]
        for i in range(num_images):
            im=cv2.imread(roidb[i]['image'])
            #将im放缩到固定尺寸
            #未完成
            height=im.shape[0]
            width=im.shape[1]
            if height>max_height:
                max_height=height
            if width>max_width:
                max_width=width
            ims.append(im)
        for i in range(num_images):
            height=ims[i].shape[0]
            width=ims[i].shape[1]
            im=cv2.resize(ims[i],(max_width,max_height),interpolation=cv2.INTER_CUBIC)
            processed_ims.append(im)
            #对gt_box进行scale
            scale=np.array([width,height])/np.array([max_width,max_height])
            boxes=np.array(roidb[i]['boxes'])#形如[box_num,4]
            boxs_scale=self.box_scale(boxes,scale)
            boxs_scale=boxs_scale.tolist()
            gt_boxes_reshape.append(boxs_scale)
        blob=self.im_list_to_blob(processed_ims)
        ##blob中只有一个图像的数据
        return blob,gt_boxes_reshape#boxes_scale为list
        
    def im_list_to_blob(self,ims):
        ##因为这里假定batch=1,所以这里的代码要精简
        max_shape=np.array([im.shape for im in ims]).max(axis=0)
        num_images=len(ims)
        blob=np.zeros((num_images,max_shape[0],max_shape[1],3),dtype=np.float32)
        for i in range(num_images):
            im=ims[i]
            blob[i,0:im.shape[0],0:im.shape[1],:]=im
        return blob
    def box_scale(self,boxes,scale):#boxes,scale均为np类型，且为浮点类型
        boxes_num=len(boxes)
        scales=np.tile(scale,[boxes_num,2])
        boxes_scale=boxes/scales
        return boxes_scale
        ##blob中只有一个图像
if __name__ =='__main__':
    myimdb=imdb_load()
    print(myimdb.roidb[0])
    for i in range(10):
        blob=myimdb._get_next_minibatch()
        print("第",i,"轮",'---',blob['im_name'])
    blob=myimdb._get_next_minibatch()
    data=blob['data'][0]
    print(data)
    #im=PIL.Image.fromarray(data)
    #im.save("out.jpg")
    cv2.imwrite("out.jpg",data)
    im=cv2.imread('out.jpg')
    boxes=blob['gt_boxes']
    #cv2.polylines(im,np.array(boxes),1,255)
    boxes=(boxes)
    for i in range(len(boxes[0])):
        cv2.rectangle(im,(int(boxes[0][i][0]),int(boxes[0][i][1])),(int(boxes[0][i][2]),int(boxes[0][i][3])),(0,0,255),2)
    plt.imshow(im)
    plt.show()
    print(blob)

