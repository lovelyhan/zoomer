# coding=utf-8
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

import tensorflow as tf



# predictions = tf.where(tf.is_nan(predictions), tf.ones_like(predictions) * 0.5, predictions)




#
#
#
#
############################################# get tensor ###############################################
#
#
def get_tensor_norm(tensor, axis, name=None):
    return tf.reduce_sum(tf.square(tensor), axis, False)

def get_padded_tensor(tensor, paddings, mode="CONSTANT", constant_values=0, name=None):
    tensor_pad = tf.pad(tensor, paddings, constant_values=constant_values)
    return tensor_pad

def get_cliped_tensor(tensor, clip_value_min, clip_value_max):
    return tf.clip_by_value(tensor, clip_value_min=clip_value_min, clip_value_max=clip_value_max)





#
#
#
#
############################################# distance, norm ###############################################
#
#
def cosine_fun(ays_src, ays_dst):
    src_norm = tf.sqrt(tf.reduce_sum(tf.square(ays_src), 1, True))
    dst_norm = tf.sqrt(tf.reduce_sum(tf.square(ays_dst), 1, True))

    prod = tf.reduce_sum(tf.multiply(ays_src, ays_dst), 1, True)
    norm_prod = tf.multiply(src_norm, dst_norm)

    cosine = tf.truediv(prod, norm_prod)
    return cosine

def get_normed_tensor(tensor, axis=-1, norm_type="l2", name=None):

    # l2 norm: output = x / sqrt(max(sum(x**2), epsilon))
    # output = tf.nn.l2_normalize(tensor, -1)
    # output = tf.truediv(tensor, tf.sqrt(tf.reduce_sum(tf.square(tensor), -1, keep_dims=True)) + 1e-10)
    output = tf.nn.l2_normalize(tensor, axis)

    return output

def get_inner_product(src_tensor, dst_tensor, name=None):
    return tf.reduce_sum(tf.multiply(src_tensor, dst_tensor), 1, True)








#
#
#
#
############################################# ActivationFunction ###############################################
#
#
def getActivationFunctionOp(act_name='relu'):
    if type(act_name) != str and type(act_name) != unicode:
        print('type(act_name) != str')
        return act_name
    if act_name.lower() == 'relu':
        return tf.nn.relu
    elif act_name.lower() == 'tanh':
        return tf.nn.tanh
    elif act_name.lower() == 'lrelu':
        return lambda x: tf.nn.leaky_relu(x, alpha=0.01)
    elif act_name.lower() == 'llrelu':
        return lambda x: tf.nn.leaky_relu(x, alpha=0.1)
    elif act_name.lower() == 'gelu':
        return lambda x: gelu(x)
    elif act_name.lower() =="elu":
        return lambda x: tf.nn.elu(x)
    else:
        return tf.nn.relu

def gelu(input_tensor):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        input_tensor: float Tensor to perform activation.
    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf

