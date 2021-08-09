# coding=utf-8

import tensorflow as tf

from model.layers import dnn_layers
from xdl.utils import tfprint


class DSSMLayer(object):
    """
    input: embedding or embedding list
    """
    def __init__(self, layer_name='DSSMLayer', inputs_or_list=None, sub_layer_num = 3,
                 input_dim=None, mid_dim=None, output_dim=None, init_type=None, init_para=None):
        self.layer_name = layer_name
        self.sub_layer_num = sub_layer_num
        assert self.sub_layer_num >= 1, "ERROR: DSSMLayer<{}>.sub_layer_num={} must >=1".format(self.layer_name, self.sub_layer_num)
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.output_dim = output_dim
        self.init_type = init_type
        self.init_para = init_para

        if self.mid_dim == None:
            self.mid_dim = self.output_dim

        self.sub_layers = None
        self.build_layer()
        # if inputs_or_list != None:
        #     self.init_outputs = self.inference(inputs_or_list)

    def build_layer(self):
        self.sub_layers = []
        output_dim_list = [self.input_dim]
        for _ in range(self.sub_layer_num-1):
            output_dim_list.append(self.mid_dim)
        output_dim_list.append(self.output_dim)
        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            print("variable_scope: " + tf.get_variable_scope().name + "    ,  reuse: ", tf.get_variable_scope().reuse)
            for idx in range(1, len(output_dim_list)):
                _layer = dnn_layers.DnnSingleLayer(layer_name="layer" + str(idx),
                                                   input_dim=output_dim_list[idx - 1], output_dim=output_dim_list[idx],
                                                   weight_init_type=self.init_type, weight_init_para=self.init_para)
                self.sub_layers.append(_layer)

    def inference(self, inputs_or_list):
        """
        Returns: logits
        """
        tfprint.simple_print("DSSMLayer", "{}.inference()".format(self.layer_name))
        if isinstance(inputs_or_list, list):
            outputs = tf.concat(inputs_or_list, 1)
        else:
            outputs = inputs_or_list
        for idx in range(self.sub_layer_num):
            outputs = self.sub_layers[idx].inference(outputs)
        return outputs











