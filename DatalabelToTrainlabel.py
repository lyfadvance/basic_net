import numpy as np
import numpy.random as npr
DTYPE=np.float
##这个函数是用来将标注的label转化成训练时的label
##通过tf.py_func的形式进行调用
'''
def bbox_overlaps(#水平的box
     boxes,
     query_boxes):

    N=boxes.shape[0]
    K=query_boxes.shape[0]
    overlaps=np.zeros((N,K),dtype=DTYPE)
    for k in range(K):
        box_area=(
            (query_boxes[k,2]-query_boxes[k,0]+1)*
            (query_boxes[k,3]-query_boxes[k,1]+1)
        )
        for n in range(N):
            iw=(
                min(boxes[n,2],query_boxes[k,2])-
                max(boxes[n,0],query_boxes[k,0])+1
            )
            if iw>0:
                ih=(
                    min(boxes[n,3],query_boxes[k,3])-
                    max(boxes[n,1],query_boxes[k,1])+1
                )
                if ih>0:
                    ua=float(
                        (boxes[n,2]-boxes[n,0]+1)*
                        (boxes[n,3]-boxes[n,1]+1)+
                        box_area-iw*ih
                    )
                    overlaps[n,k]=iw*ih/ua
    return overlaps
'''
def compute_box_area(box):#box形如[len=4]或[len=8]
    if len(box)==4:
        return (box[2]-box[0]+1)*(box[3]-box[1]+1)
    elif len(box)==8:
        v1=box[0:2]-box[2:4]
        v2=box[4:6]-box[6:8]

        return np.linalg.norm(v2)*np.linalg.norm(v1)
def compute_box_overlaps(anchor,gt_box):#anchor形如[len=4],gt_box形如[len=8].不采用IOU的形式.因为anchor和gt_box的差距很大.计算overlap/anchor的大小
    #计算重叠的像素值有点困难，现在实现重叠返回1,不重叠返回0
    
    #计算gt_box的中心
    gt_box_v1=gt_box[0:2]
    gt_box_v3=gt_box[4:6]
    
    anchor_v1=anchor[0:2]
    anchor_v2=anchor[2:4]

    gt_box_center=(gt_box_v3+gt_box_v1)/2
    anchor_center=(anchor_v2+anchor_v1)/2
    
    #计算anchor和gt_box的偏移向量
    v_offset=anchor_center-gt_box_center
    #print("anchor偏移",v_offset)
    #计算gt_box的短轴向量
    gt_box_v2=gt_box[2:4]
    gt_box_v2_v3_center=(gt_box_v2+gt_box_v3)/2
    if gt_box_v2_v3_center[1]>=gt_box_center[1]:#保证向量角度在0-180
        v_axis=gt_box_v2_v3_center-gt_box_center
    else:
        v_axis=gt_box_center-gt_box_v2_v3_center
    #print("短轴向量",v_axis)
    #计算anchor中心相对于gt_box中心在短轴上的偏移
    cos_value=np.abs(np.sum(v_offset*v_axis)/(np.linalg.norm(v_offset)*np.linalg.norm(v_axis)))
    v_heighth_offset=np.linalg.norm(v_offset)*cos_value
    v_width_offset=np.linalg.norm(v_offset)*np.sqrt(1-cos_value*cos_value)
    #print("高度偏移",v_heighth_offset) 
    #计算gt_box左上角相对于gt_box中心在短轴上的偏移
    gt_v_offset=gt_box_v1-gt_box_center

    gt_cos_value=np.abs(np.sum(gt_v_offset*v_axis)/(np.linalg.norm(gt_v_offset)*np.linalg.norm(v_axis)))
    gt_v_heighth_offset=np.linalg.norm(gt_v_offset)*gt_cos_value
    gt_v_width_offset=np.linalg.norm(gt_v_offset)*np.sqrt(1-gt_cos_value*gt_cos_value)
    #print("gt高度偏移",gt_v_heighth_offset) 
    #计算anchor_center是否在gt_box里
    if v_width_offset<=gt_v_width_offset and v_heighth_offset<=gt_v_heighth_offset:
        label=1
        #计算回归值
        '''
        if np.sum(v_axis*v_offset)>0:#即偏移同向
            dy=-v_heighth_offset
        else:
            dy=v_heighth_offset
        #计算angle
        cos_angle=v_axis[0]/np.linalg.norm(v_axis)
        #print(cos_angle)
        angle=np.arccos(cos_angle)
        angle=angle/(np.pi*2)*360
        #计算高度
        dh=np.linalg.norm(v_axis)
        print(label,dy,dh,angle)
        '''
    else:
        
        label=0
        '''
        dh=0
        dy=0
        angle=0
        '''
    #return np.array([label,dh,dy,angle])
    return label
    
        


def bbox_overlaps(anchors,gt_boxes):#,anchors形如[[len=4]..[]],gt_boxes形如[[len=8]..[]]
    K=anchors.shape[0]
    N=gt_boxes.shape[0]
    overlaps=np.zeros((K,N),dtype=DTYPE)
    for k in range(K):
        for n in range(N):
            overlaps[k,n]=compute_box_overlaps(anchors[k],gt_boxes[n])
    return overlaps #overlaps形如[[]..[]],shape为[len(anchors),len(gt_boxes),4]
def compute_box_angle_height(box):#box形如[len=8]
    return angle,height
def DatalabelToTrainlabel_layer(rpn_cls_score,gt_boxes,im_info):
    _anchors=np.array([[0,0,15,15]],np.float32)
    _num_anchors=_anchors.shape[0]
    #根据特征图构造所有的anchor
    #anchor的顺序是按照从上到下，从左到右一次展开
    height,width=rpn_cls_score.shape[1:3]
    K=height*width
    A=_num_anchors
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
    
                        
    #all_anchors=[] #(K*A,4)
    total_anchors=int(K*A)
    #仅保留那些还在图像内部的anchor，超出图像的都删掉
    _allowed_border=0
    #因为一批中只有一张图像
    im_info=im_info[0]
    #print("________________________________________label转换执行结束")
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]
    
    anchors=all_anchors[inds_inside,:]
    #计算label
    labels=np.empty((len(inds_inside),),dtype=np.float32)
    labels.fill(-1)
    
    overlaps=bbox_overlaps(
        np.ascontiguousarray(anchors,dtype=np.float),
        np.ascontiguousarray(gt_boxes,dtype=np.float))
    #print("overlaps",overlaps)
#找到每个anchor对应的overlap最大的gt_box
    argmax_overlaps=overlaps.argmax(axis=1)
    max_overlaps=overlaps[np.arange(len(inds_inside)),argmax_overlaps]
#找到每个gt_box对应的overlap最大的anchor
    gt_argmax_overlaps=overlaps.argmax(axis=0)
    gt_max_overlaps=overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
    #背景设置为0
    labels[max_overlaps<0.3]=0 
    #每个gt_box所对应的overlap最大的anchor设置为1
    labels[gt_argmax_overlaps]=1
    #每个anchor的最大overlap大于0.7设置为1
    labels[max_overlaps>0.7]=1

#对所有的label进行采样
    #对正样本进行采样，使其个数在128个一下
    fg_inds=np.where(labels==1)[0]
    if len(fg_inds)>128:
        disable_inds=npr.choice(
            fg_inds,size=(len(fg_inds)-128),replace=False)
        labels[disable_inds]=-1
    #对负样本进行采样，使其个数在256个以下
    bg_inds=np.where(labels==0)[0]
    if len(bg_inds)>256:
        disable_inds=npr.choice(
            bg_inds,size=(len(bg_inds)-256),replace=False)
        labels[disable_inds]=-1
    
    #计算权重
    #bbox_inside_weights=np.zeros((len(inds_inside),4),dtype=np.float32)
    #bbox_inside_weights[labels==1,:]=np.array([0,1,0,1])
    bbox_outside_weights=np.zeros((len(inds_inside),3),dtype=np.float32)
    positive_weights = np.ones((1, 3))
    negative_weights = np.zeros((1, 3))
    bbox_outside_weights[labels==1,:]=positive_weights
    bbox_outside_weights[labels==0,:]=negative_weights
    #bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)#内部权重以0填充
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)#外部权重以0填充

    # bbox_inside_weights
    #bbox_inside_weights = bbox_inside_weights \
    #    .reshape((1, height, width, A * 4))

    #rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 3))
    rpn_bbox_outside_weights = bbox_outside_weights

    ##计算score label和边框回归的label
    bbox_targets=np.zeros((len(inds_inside),3),dtype=np.float32)
    bbox_targets=_compute_targets(anchors,gt_boxes[argmax_overlaps,:])
    labels=_unmap(labels,total_anchors,inds_inside,fill=-1)
    labels=labels.reshape((1,height,width,A))
    rpn_labels=labels

    bbox_targets=_unmap(bbox_targets,total_anchors,inds_inside,fill=0)
    bbox_targets=bbox_targets.reshape((1,height,width,A*3))
    rpn_bbox_targets=bbox_targets

    return rpn_labels,rpn_bbox_targets,rpn_bbox_outside_weights
#因为之前将一些anchor样本舍去了，这里重新构建所有的anchor,只不过舍去的anchor设置为不感兴趣,dont care
def _unmap(data,count,inds,fill=0):#data指label的数据，count:所有的anchor的数量,inds:保留的anchor的坐标
    #对label进行扩充
    if len(data.shape)==1:
        ret =np.empty((count,),dtype=np.float32)
        ret.fill(fill)
        ret[inds]=data
    #对边框回归进行扩充
    else:
        ret=np.empty((count,)+data.shape[1:],dtype=np.float32)
        ret.fill(fill)
        ret[inds,:]=data
    return ret



def _compute_targets(ex_rois, gt_rois):#gt_rois[i]是所有gt_box中，与ex_rois[i](anchor[i])重叠最大的gt_box
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 8

    return bbox_transform(ex_rois, gt_rois[:, :8]).astype(np.float32, copy=False)

###########################################
#计算边框回归的label
###########################################
def compute_box_regress(anchor,gt_box):#anchor形如[len=4],gt_box形如[len=8].不采用IOU的形式.因为anchor和gt_box的差距很大.计算overlap/anchor的大小
    #计算重叠的像素值有点困难，现在实现重叠返回1,不重叠返回0
    
    #计算gt_box的中心
    gt_box_v1=gt_box[0:2]
    gt_box_v3=gt_box[4:6]
    
    anchor_v1=anchor[0:2]
    anchor_v2=anchor[2:4]

    gt_box_center=(gt_box_v3+gt_box_v1)/2
    anchor_center=(anchor_v2+anchor_v1)/2
    
    #计算anchor和gt_box的偏移向量
    v_offset=anchor_center-gt_box_center
    #print("anchor偏移",v_offset)
    #计算gt_box的短轴向量
    gt_box_v2=gt_box[2:4]
    gt_box_v2_v3_center=(gt_box_v2+gt_box_v3)/2
    if gt_box_v2_v3_center[1]>=gt_box_center[1]:#保证向量角度在0-180
        v_axis=gt_box_v2_v3_center-gt_box_center
    else:
        v_axis=gt_box_center-gt_box_v2_v3_center
    #print("短轴向量",v_axis)
    #计算anchor中心相对于gt_box中心在短轴上的偏移
    cos_value=np.abs(np.sum(v_offset*v_axis)/(np.linalg.norm(v_offset)*np.linalg.norm(v_axis)))
    v_heighth_offset=np.linalg.norm(v_offset)*cos_value
    v_width_offset=np.linalg.norm(v_offset)*np.sqrt(1-cos_value*cos_value)
    #print("高度偏移",v_heighth_offset) 
    #计算gt_box左上角相对于gt_box中心在短轴上的偏移
    gt_v_offset=gt_box_v1-gt_box_center

    gt_cos_value=np.abs(np.sum(gt_v_offset*v_axis)/(np.linalg.norm(gt_v_offset)*np.linalg.norm(v_axis)))
    gt_v_heighth_offset=np.linalg.norm(gt_v_offset)*gt_cos_value
    gt_v_width_offset=np.linalg.norm(gt_v_offset)*np.sqrt(1-gt_cos_value*gt_cos_value)
    #print("gt高度偏移",gt_v_heighth_offset) 
    #计算anchor_center是否在gt_box里
    if v_width_offset<=gt_v_width_offset and v_heighth_offset<=gt_v_heighth_offset:
        label=1
        #计算回归值
        
        if np.sum(v_axis*v_offset)>0:#即偏移同向
            dy=-v_heighth_offset
        else:
            dy=v_heighth_offset
        #计算angle
        cos_angle=v_axis[0]/np.linalg.norm(v_axis)
        #print(cos_angle)
        angle=np.arccos(cos_angle)
        angle=angle/(np.pi*2)*360
        #计算高度
        dh=np.linalg.norm(v_axis)
        print(label,dy,dh,angle)
        
    else:
        assert 1!=1,'计算回归的时候,label!=1'
        label=0
        '''
        dh=0
        dy=0
        angle=0
        '''
    #return np.array([label,dh,dy,angle])
    return np.array([dh,dy,angle])
def bbox_transform(ex_rois,gt_rois):#ex_rois形如[[len=4]..[]],gt_rois形如[[len=8]...[]]
    #targets_dy高度回归值,形如[len=len(ex_rois)]
    #targets_dh高度回归值,形如[len=len(ex_rois)]
    #targets_angle,角度回归值,形如[len=len(ex_rois)]
    '''
    targets_dy=np.zeros((len(ex_rois),))
    targets_dh=np.zeros((len(ex_rois),))
    targets_angle=np.zeros((len(ex_rois),))
    '''
    targets=np.zeros((len(ex_rois),3))
    for i in range(len(ex_rois)):
        target=compute_box_regress(ex_rois[i],gt_rois[i])
        targets[i]=target
    return targets
    #targets=np.vstack(targets_dy,targets_dh,targets_angle).transpose()
'''
def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
        'Invalid boxes found: {} {}'. \
            format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    #为了与fast rcnn兼容，计算了4个回归，实际上只需要两个回归
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets
'''
'''
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
'''
'''
def box_to_normalform(box):#将box的坐标转化为标准形式,box的形式为按照圆周旋转，也不是任意的
    v1=box[0:2]
    v2=box[2:4]
    v3=box[4:6]
    v4=box[6:8]
    start=0
    v1_v2=np.linalg.norm(v2-v1)
    v2_v3=np.linalg.norm(v3-v2)
    if v2_v3>v1_v2:#v1_v2为短边
        if v1[0]+v2[0]>v3[0]+v4[0]:#v1_v2在左边
            if v2[1]>v1[0]:#v2在上边
                start=2
            else :
                start=1
        else:
            if
'''
def compute_box_angle(box):#box形如[左上,左下,右上，右下],且左上是,以长宽中长为方向,且从左指向右的方向的左上
    x1=box[0]
    y1=box[1]
    x2=box[2]
    y2=box[3]
    #计算角度和高度
    return angle,height
if __name__=='__main__':
    box=np.array([[0,31,0,0,63,0,63,31]])
    anchor=np.array([[0,0,15,15],[0,16,15,31]])
    #compute_box_overlaps(anchor,box)
    overlaps=bbox_overlaps(anchor,box)
    print(overlaps)
    
    boxs=np.array([[0,31,0,0,63,0,63,31],[0,31,0,0,63,0,63,31]])
    results=_compute_targets(anchor,boxs)
    print(results)
#############还没有测试边框回归的计算,测试了score的计算
