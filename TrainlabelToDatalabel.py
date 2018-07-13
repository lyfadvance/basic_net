import numpy as np
import numpy.random as npr

def TrainlabelToDatalabel_layer(rpn_cls_prob_reshape,im_info):
    _num_anchors=1
    height,width=rpn_cls_prob_reshape.shape[1:3]
    print("______________________",height,width)
    scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:,:,:,:,1],
                        [1, height, width, _num_anchors])
    scores=np.reshape(scores,[height,width])
    return scores
