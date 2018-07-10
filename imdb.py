import os
import os.path as osp
import PIL
import numpy as np
import scipy.sparse

class imdb(object):
    def __init__(self,name):
        self._name=name
        self._num_classes=2
        self._classes=[]
        self._image_index=[]
        self._roidb=None

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)
    
    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index
    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self,val):
        self._roidb_handler=val

    @property
    def roidb(self):
        if self._roidb is not None:
            return self._roidb
        self._roidb=self.roidb_handler()
        return self._roidb

    @property
    def num_images(self):
        return len(self.image_index)
