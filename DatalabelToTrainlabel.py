import numpy as np
import numpy.random as npr
DTYPE=np.float
##这个函数是用来将标注的label转化成训练时的label
##通过tf.py_func的形式进行调用
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
def DatalabelToTrainlabel_layer(rpn_cls_score,gt_boxes,im_info):
p   _anchors=np.array([[0,0,15,15]],np.int32)
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

    ##计算score label和边框回归的label
    bbox_targets=np.zeros((len(inds_inside),4),dtype=np.float32)
    bbox_targets=_compute_targets(anchors,gt_boxes[argmax_overlaps,:])

    labels=_unmap(labels,total_anchors,inds_inside,fill=-1)
    labels=labels.reshape((1,height,width,A))
    rpn_labels=labels
    return rpn_labels
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
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


###########################################
#计算边框回归的label
###########################################
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
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets
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
if __name__=='__main__':
    rpn_cls_score=np.zeros((1,50,45,3))
    im_info=np.array([[800,720]])
    gt_boxes=np.array([[16,0,31,15,1]])
    labels=anchor_target_layer(rpn_cls_score,gt_boxes,im_info)
    print(labels.reshape(50,45))
    labels=labels.reshape(50,45)
    s=0
    for i in range(50):
        for j in range(45):
            if labels[i,j]==0:
                s=s+1
    print(s)

    gt_boxes=np.array([[5,0,18,18,1]])
    labels=anchor_target_layer(rpn_cls_score,gt_boxes,im_info)
    print(labels.reshape(50,45))
#############还没有测试边框回归的计算,测试了score的计算
