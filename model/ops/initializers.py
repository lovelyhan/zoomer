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



def get_initializer(init_type, init_para = None,
                    # dtype=tf.float32, mean=0.0, stddev=0.1, value=0.0, seed=None, factor=1.0, minval=0.0, maxval=1.0
                    ):
    if init_type == None:
        init_type = 1
    if init_para == None:
        init_para = {}
    dtype = init_para.get("dtype", tf.float32)
    seed = init_para.get("seed", None)

    if init_type == 0:
        value = init_para.get("value", 0.0)
        return tf.constant_initializer(value=value, dtype=dtype)
    elif init_type == 1:
        mean = init_para.get("mean", 0.0)
        stddev = init_para.get("stddev", 0.1)
        return tf.truncated_normal_initializer(mean=mean, stddev=stddev, dtype=dtype, seed=seed)
    elif init_type == 2:
        factor = init_para.get("factor", 1.0)
        return tf.uniform_unit_scaling_initializer(factor=factor, seed=seed, dtype=dtype)
    elif init_type == 3:
        minval = init_para.get("minval", 0.0)
        maxval = init_para.get("maxval", 1.0)
        return tf.random_uniform_initializer(minval=minval, maxval=maxval)
    elif init_type == 4:
        # 正交分布
        return tf.orthogonal_initializer(gain=1.0, seed=seed, dtype=dtype)
    elif init_type == 5:
        return tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
    elif init_type == 6:
        # TODO
        return tf.contrib.layers.variance_scaling_initializer()
    else:
        return None


def get_Orthogonal_Matrix(shape, dtype=tf.float32, gain=1.0, seed=None):
    from tensorflow.python.ops import init_ops_v2
    # returned tensor W s.t.: W^T * W = I, but W * W^T != I  (can be transposed)
    # shape = [m,n ], if m>=n, then ret is left orthogonal (W^T * W = I); if m < n, then ret is right orthogonal
    ##############
    # pinv:
    # AA = tf.constant([[1., 0.4, 0.5, 0.8, 0.8],
    #                   [0.4, 0.2, 0.25, 0.7, 0.8],
    #                   [0.5, 0.25, 0.35, 0.6, 0.7]], dtype=tf.float32)
    # AA_left = tf.matmul(tf.linalg.pinv(AA), AA)  #  != I
    # AA_right = tf.matmul(AA, tf.linalg.pinv(AA))  #  == I
    return init_ops_v2.Orthogonal(gain=gain, seed=seed).__call__(shape=shape, dtype=dtype)




def get_variable_with_initialize(name, shape, dtype=tf.float32, trainable=True, init_type = None, init_para=None,
                                 regularizer=None):
    # var = tf.get_variable(name,
    #                       shape=shape,
    #                       dtype=dtype,
    #                       trainable=None,
    #                       initializer=None,
    #                       regularizer=tf.nn.l2_loss)
    if init_para == None:
        init_para = {}
    init_para["dtype"] = dtype
    var = tf.get_variable(name,
                          shape = shape,
                          dtype = dtype,
                          trainable=trainable,
                          initializer=get_initializer(init_type, init_para),
                          regularizer=regularizer)
    print("get_variable: " + var.name + "\t shape: {}".format(shape) +
          "\t dtype: {}".format(dtype) +
          "\t trainable: {}".format(trainable) +
          "\t init_type: {}".format(init_type) +
          "\t init_para: {}".format(init_para) +
          "\t regularizer: {}".format(regularizer))
    return var
