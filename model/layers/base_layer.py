# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# coding=utf-8

import tensorflow as tf
from model.ops import tf_ops

class BaseLayer(object):
    def __init__(self, layer_name='BaseLayer', layer_config=None):

        self.layer_name = layer_name
        self.layer_config = layer_config

        """ A float32 Tensor with shape [batch_size, -1]. """
        self.layer_input = None

        """ A float32 Tensor with shape [batch_size, -1]. """
        self.layer_ouput = None

        # self.weights = None
        # self.biases = None
        # self.activation_name = None

    def inference(self, inputs):
        """
        Returns: logits
        """
        return None

    def activation_func(self, inputs, act_name="elu"):
        """
        Returns: logits
        """
        if act_name == None or act_name == "":
            print("layer_name={}  ->  activation_func is None (i.e. No activation)".format(self.layer_name))
            return inputs

        act_name = act_name.lower()
        if act_name == 'relu':
            return tf.nn.relu(inputs)
        elif act_name == "elu":
            # print("layer_name={}  ->  activation_func: tf.nn.elu".format(self.layer_name))
            return tf.nn.elu(inputs)
        elif act_name == "tanh":
            return tf.nn.tanh(inputs)
        elif act_name == "sigmoid":
            return tf.nn.sigmoid(inputs)
        elif act_name == "lrelu":
            return tf.nn.leaky_relu(inputs, alpha=0.01)
        elif act_name == "llrelu":
            return tf.nn.leaky_relu(inputs, alpha=0.1)
        elif act_name == "gelu":
            return tf_ops.gelu(inputs)
        else:
            raise ValueError("act_name={} unvaild!".format(act_name))

