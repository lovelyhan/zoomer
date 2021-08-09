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
from base_layer import BaseLayer
from model.ops import initializers


class DnnSingleLayer(BaseLayer):
    def __init__(self, layer_name='DnnSingleLayer', input_dim=None, output_dim=None,
                 weight_init_type=None, weight_init_para=None,
                 has_biases = True, bias_init_type=None, bias_init_para=None,
                 act_name="elu", regularizer=tf.nn.l2_loss):
        super(DnnSingleLayer, self).__init__(layer_name=layer_name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init_type = weight_init_type
        self.weight_init_para = weight_init_para
        self.has_biases = has_biases
        self.bias_init_type = 0 if bias_init_type is None else bias_init_type
        self.bias_init_para = {"value": 0.0} if bias_init_para is None else bias_init_para
        self.act_name = act_name
        self.regularizer = regularizer

        self.build_layer()

    def build_layer(self):
        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            print("variable_scope: " + tf.get_variable_scope().name + "    ,  reuse: ", tf.get_variable_scope().reuse)
            self.weights = initializers.get_variable_with_initialize(
                "weights", [self.input_dim, self.output_dim],
                init_type=self.weight_init_type, init_para=self.weight_init_para,
                regularizer = self.regularizer)
            if self.has_biases:
                self.biases = initializers.get_variable_with_initialize(
                    "biases", [self.output_dim],
                    init_type=self.bias_init_type, init_para=self.bias_init_para,
                    regularizer=None)

    def inference(self, inputs):
        """
        Returns: logits
        """
        layer1_output0 = tf.matmul(inputs, self.weights)
        if self.has_biases:
            layer1_output0 = layer1_output0 + self.biases
        layer1_output = self.activation_func(layer1_output0, act_name = self.act_name)
        return layer1_output



