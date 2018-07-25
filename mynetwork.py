import numpy as np
import tensorflow as tf
from DatalabelToTrainlabel import DatalabelToTrainlabel_layer
from TrainlabelToDatalabel import TrainlabelToDatalabel_layer
WEIGHT_DECAY=0.0005
DEFAULT_PADDING='SAME'
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        print("----------------------------name",name)
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path,encoding='latin1').item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model "+subkey+ " to "+key)
                    except ValueError:
                        print("ignore "+key)
                        if not ignore_missing:

                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in list(self.layers.items()))+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')
    
    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias, name=scope.name)
                return tf.nn.bias_add(conv, biases, name=scope.name)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv, name=scope.name)
                return conv
    @layer
    def abs_conv(self,input,k_h,k_w,c_o,s_h,s_w,name,biased=True,relu=True,padding=DEFAULT_PADDING,trainable=True):
        self.validate_padding(padding)
        c_i=input.get_shape()[-1]
        convolve=lambda i,k:tf.nn.conv2d(i,k,[1,s_h,s_w,1],padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights=tf.truncated_normal_initializer(0.0,stddev=0.01)
            init_biases=tf.constant_initializer(0.0)
            kernel=self.make_var('weights',[k_h,k_w,c_i,c_o],init_weights,trainable,\
                                    regularizer=self.l2_regularizer(WEIGHT_DECAY))
            if biased:
                biases=self.make_var('biases',[c_o],init_biases,trainable)
                conv=convolve(input,kernel)
                if relu:
                    bias=tf.abs(tf.nn.bias_add(conv,biases))
                    return 1-tf.nn.relu(bias,name=scope.name)
                return tf.abs(tf.nn.bias_add(conv,biases,name=scope.name))
            else:
                print("-------------------error")
    @layer
    def concat(self, inputs, axis, name):
        with tf.variable_scope(name) as scope:
            return tf.concat(axis=axis, values=inputs, name=scope.name)

    @layer
    def regress(self,input,d_i,d_o,name,trainable=True):
        with tf.variable_scope(name) as scope:
            shape=tf.shape(input)
            N,H,W,C=shape[0],shape[1],shape[2],shape[3]
            input=tf.reshape(input,[N*H*W,C])

            init_weights=tf.truncated_normal_initializer(0.0,stddev=0.01)
            init_biases=tf.constant_initializer(0.0)
            kernel=self.make_var('weights',[d_i,d_o],init_weights,trainable,regularizer=self.l2_regularizer(WEIGHT_DECAY))
            biases=self.make_var('biases',[d_o],init_biases,trainable)

            _O=tf.matmul(input,kernel)+biases
            return tf.reshape(_O,[N,H,W,int(d_o)])
    
    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def DatalabelToTrainlabel_layer(self,input,name):
        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
                tf.py_func(DatalabelToTrainlabel_layer,
                           [input[0],input[1],input[2]],
                           [tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels') # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets') # shape is (1 x H x W x A, 4)
            #rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights') # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights') # shape is (1 x H x W x A, 4)


            return rpn_labels, rpn_bbox_targets, rpn_bbox_outside_weights
    @layer
    def TrainlabelToDatalabel_layer(self,input,name):
        with tf.variable_scope(name) as scope:
            rpn_rois=tf.py_func(TrainlabelToDatalabel_layer,[input[0],input[1],input[2]],[tf.float32])
            rpn_rois=rpn_rois[0]
            rpn_rois=tf.convert_to_tensor(rpn_rois,name='rpn_rois')
            self.layers['rpn_rois']=rpn_rois
            return rpn_rois
    '''
    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer_py,
                           [input[0],input[1],input[2],input[3],input[4], _feat_stride, anchor_scales],
                           [tf.float32,tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels') # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets') # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights') # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights') # shape is (1 x H x W x A, 4)


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    '''
    '''
    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)
    @layer
    def batch_normalization(self,input,name,relu=True,is_training=False):
        """contribution by miraclebiu"""
        if relu:
            temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            return tf.nn.relu(temp_layer)
        else:
            return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)
    '''
    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input,\
                               [input_shape[0],\
                                input_shape[1], \
                                -1,\
                                int(d)])

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                #return tf.mul(l2_weight, tf.nn.l2_loss(tensor), name='value')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

if __name__=='__main__':
    print('test')
        
