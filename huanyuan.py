import numpy as np
import xml.etree.ElementTree as ET
import os
import os.path as osp
#判断是否可以连接
ROOT_DIR=osp.abspath(osp.join(osp.dirname(__file__)))
DATA_DIR=osp.abspath(osp.join(ROOT_DIR))
def _get_data_path():
        return os.path.join(DATA_DIR,'VOCdevkit2007','VOC2007')
_data_path=_get_data_path()
def _load_image_set_index():
    image_set_file=os.path.join(_data_path,'ImageSets','Main','train.txt')
    assert os.path.exists(image_set_file),'Path does not exist:{}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index=[x.strip() for x in f.readlines()]
    return image_index
def _load_pascal_annotation(index):
        filename=os.path.join(_data_path,'Annotations',index+'.xml')
        tree=ET.parse(filename)
        objs=tree.findall('object')
        num_objs=len(objs)

        boxes=np.zeros((num_objs,4),dtype=np.uint16)
        #gt_classes=np.zeros((num_objs),dtype=np.int32)
        
        #overlaps=np.zeros((num_objs,self.num_classes),dtype=np.float32)
        #box的面积
        #seg_areas=np.zeros((num_objs),dtype=np.float32)

        for ix,obj in enumerate(objs):
            bbox=obj.find('bndbox')
            ##水平的
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            boxes[ix,:]=[x1,y1,x2,y2]
        return boxes

        
def judge(chain,box):
    for i in range(len(chain)):
        if box[1]==chain[i][0][1] and box[3]==chain[i][0][3]:
            return i
    else:
        return -1
def chain_merge(chain):
    new_chain=[]
    for i in range(len(chain)):
        li=chain[i]
        x_min=100000
        x_max=-1
        for j in range(len(li)):
            box=li[j]
            if box[0]<x_min:
                x_min=box[0]
            if box[2]>x_max:
                x_max=box[2]
        new_chain.append([x_min,li[0][1],x_max,li[0][3]])
    return new_chain
def huanyuan(boxes):
    chain=[]
    for i in range(len(boxes)):
        if len(chain)==0:
            #print(chain)
            chain.append([boxes[i]])
        else:
            index=judge(chain,boxes[i])
            if index==-1:
                chain.append([boxes[i]])
            else:
                chain[index].append(boxes[i])
    #print(chain)
    new_chain=chain_merge(chain)
    #print(new_chain)
    return new_chain

if __name__=='__main__':
    image_index=_load_image_set_index()
    roidb=[_load_pascal_annotation(index) for index in image_index]
    for i in range(len(roidb)):
        new_chain=huanyuan(roidb[i])
        with open('annotations/'+image_index[i]+'.txt','w') as f:
            for j in range(len(new_chain)):
                f.write(str(new_chain[j][0]))
                f.write(',')
                f.write(str(new_chain[j][1]))
                f.write(',')
                f.write(str(new_chain[j][2]))
                f.write(',')
                f.write(str(new_chain[j][3]))
                f.write('\n')

    #print(roidb[0])
    #huanyuan(roidb[0])


