import os
import numpy as np
import scipy.sparse
try:
    import cPickle as pickle
except:
    import pickle

import xml.etree.ElementTree as ET
from .imdb import imdb
class imdb_load(imdb):
    def __init__(self):
        imdb.__init__(self,'dataset')
        self._classes=('background','text')
        self._data_path=_get_data_path()
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext='.jpg'
        self._image_index=self._load_image_set_index()
        self._roidb_handler=self.gt_roidb
        
        ###打乱roidb的顺序
        self._shuffle_roidb_inds()
    def image_path_at(self,i):
        return self.image_path_from_index(self._image_index[i])
    def iamge_path_from_index(self,index):
        image_path=os.path.join(self._data_path,'JPEGImages',index+self._image_ext)
        assert os.path.exists(image_path),\
                'Path does not exist:{}'.format(image_path)
        return image_path
    #找到数据集的地址
    def _get_data_path(self):
        return os.path.join(cfg.DATA_DIR,'dataset')
    def _load_image_set_index(self):
        image_set_file=os.path.join(self._data_path,'ImageSets','Main',self._image_set+'.txt')
        assert os.path.exists(image_set_file),'Path does not exist:{}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index=[x.strip() for x in f.readlines()]
        return image_index
##准备roidb
    def gt_roidb(self):
        gt_roidb=[self._load_pascal_annotation(index) for index in self.image_index]
        return gt_roidb
    def _load_pascal_annotation(self,index):
        filename=os.path.join(self._data_path,'Annotations',index+'.xml')
        tree=ET.parse(filename)
        objs=tree.findall('object')
        num_objs=len(objs)

        boxes=np.zeros((num_objs,4),dtype=np.uint16)
        gt_classes=np.zeros((num_objs),dtype=np.int32)
        
        overlaps=np.zeros((num_objs,self.num_classes),dtype=np.float32)
        #box的面积
        seg_areas=np.zeros((num_objs),dtype=np.float32)

        for ix,obj in enumerate(objs):
            bbox=obj.find('bndbox')
            ##水平的
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            cls=self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix,:]=[x1,y1,x2,y2]
            gt_classes[ix]=cls
            overlaps[ix,cls]=1.0
            seg_areas[ix]=(x2-x1+1)*(y2-y1+1)
        overlaps=scipy.sparse.csr_matrix(overlaps)
        return {'boxes':boxes,
                'gt_classes':gt_classes,
                'gt_overlaps':overlaps,
                'seg_areas' :seg_areas}
#准备roidb
        def get_training_roidb(self):
            sizes=[PIL.Image.open(self.image_path_at(i)).size
                for i in range(self.num_images)]
            roidb=self.roidb
            for i in range(len(self.image_index)):
                roidb[i]['image']=self.image_path_at(i)
                roidb[i]['width']=sizes[i][0]
                roidb[i]['height']=sizes[i][1]
                ##这个gt_overlaps在self.roidb中已经准备好了
                gt_overlaps=roidb[i]['gt_overlaps'].toarray()
                #这个将类别矩阵转化为向量,max_overlaps其实都等于1,没什么用
                max_overlaps=gt_overlaps.max(axis=1)
                #这个将类别矩阵转化为向量，每个元素代表类别
                max_classes=gt_overlaps.argmax(axis=1)
                roidb[i]['max_classes']=max_classes
                roidb[i]['max_overlaps']=max_overlaps
##这里集成batch的获得
        def _shuffle_roidb_inds(self):
            self._perm=np.random.perutation(np.arrange(len(self._roidb)))
            self._cur=0
        def _get_next_minibatch_inds(self):
            if self._cur+1>= len(self._roidb):
                self._shuffle_roidb_inds()
            db_inds=self._perm[self._cur:self._cur+1]
            self._cur+=1
            
            return db_inds
        def _get_next_minibatch(self):
            db_inds=self._get_next_minibatch_inds()
            minibatch_db=[self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db,self._num_classes)
        def get_minibatch(roidb,num_classes):
            num_images=len(roidb)
            im_blob=_get_image_blob(roidb)
            blobs={'data':im_blob}

            assert len(roidb)==1,"Single batch only"
            #这里where的用法颇为奇怪,np.where返回的是(坐标,)所以需要[0]
            #返回的是类别是文本的box
#为兼容多类别的冗余代码
            gt_inds=np.where(roidb[0]['gt_classes']!=0)[0]
            gt_boxes=np.empty((len(gt_inds),5),dtype=np.float32)
            gt_boxes[:,0:4]=roidb[0]['boxes'][gt_inds,:]
            gt_boxes[:,4]=roidb[0]['gt_classes'][gt_inds]
            blobs['gt_boxes']=gt_boxes
            blobs['im_info']=np.array(
                [[im_blob.shape[1],im_blob.shape[2]]]
            blobs['im_name']=os.path.basename(roidb[0]['image'])
            return blobs
        def _get_image_blob(roidb):
            num_images=len(roidb)#这里其实就是等于1,为方便与ctpn的代码结合
            processed_ims=[]
            for i in range(num_images):
                im=cv2.imread(roidb[i]['image'])
                processed_ims.append(im)
            blob=im_list_to_blob(processed_ims)
            ##blob中只有一个图像的数据
            return blob
            
        def im_list_to_blob(ims):
            ##因为这里假定batch=1,所以这里的代码要精简
            max_shape=np.array([im.shape for im in ims]).max(axis=0)
            nu_images=len(ims)
            blob=np.zeros((num_images,max_shape[0],max_shape[1],3),dtype=np.float32)
            for i in range(num_images):
                im=ims[i]
                blob[i,0:im.shape[0],0:im.shape[1],:]=im
            return blob
            ##blob中只有一个图像
