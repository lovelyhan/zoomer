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
from xdl.utils import tfprint
from xdl.utils import tf_utils

def streaming_auc(predictions, labels, num_thresholds=2000, decay_rate=1):
    """
    计算的是所有单次计算的auc的累积平均值
    而且注意是近似计算，num_thresholds越大，计算越准。
    在run auc_value之前一定要先run update_op
    """
    # predictions = tf.reshape(predictions, [-1, 1])
    # labels = tf.reshape(labels, [-1, 1])
    auc_value, auc_update_op = tf.contrib.metrics.streaming_auc(predictions, labels, num_thresholds=num_thresholds)
    return auc_value, auc_update_op



####################################################################################################

def __auc(cosine, labels, num_thresholds=2000, decay_rate=1, gamma=5.0):
    predictions = tf.nn.sigmoid(cosine*gamma)
    predictions = tf.reshape(predictions, [-1])
    labels = tf.reshape(labels, [-1])
    auc_value, update_op = tf.contrib.metrics.streaming_auc(predictions, labels, num_thresholds=num_thresholds)
    return auc_value, update_op


def cosine_auc_without_labels(cosine_with_neg=None, cosine_tar=None, cosine_neg_list=None,
                              cosine_sim_type="sigmoid", sigmoid_weight=1.0,
                              num_thresholds=2000):
    tfprint.simple_print("cosine_auc_without_labels",
                         "cosine_sim_type={},sigmoid_weight={},num_thresholds={}".format(
                             cosine_sim_type, sigmoid_weight, num_thresholds))
    if cosine_with_neg != None:
        sample_num = tf.shape(cosine_with_neg)[0]
        neg_num = tf.shape(cosine_with_neg)[1] - 1
    else:
        cosine_with_neg = tf.concat([cosine_tar] + cosine_neg_list, 1)
        sample_num = tf.shape(cosine_tar)[0]
        neg_num = len(cosine_neg_list)
    sim_with_neg = tf_utils.to_sim_score(cosine_with_neg, sim_type=cosine_sim_type, sigmoid_weight=sigmoid_weight)
    predictions = tf.reshape(sim_with_neg, [-1, 1])
    labels_matrix = tf.concat([tf.ones([sample_num, 1], tf.int32), tf.zeros([sample_num, neg_num], tf.int32)], axis=1)
    labels = tf.reshape(labels_matrix, [-1, 1])
    auc_value, auc_update_op = streaming_auc(predictions, labels, num_thresholds=num_thresholds)
    return auc_value, auc_update_op

def cosine_auc_with_labels(cosines, labels, cosine_sim_type="sigmoid", sigmoid_weight=1.0, num_thresholds=2000):
    tfprint.simple_print("cosine_auc_with_labels",
                         "cosine_sim_type={},sigmoid_weight={},num_thresholds={}".format(
                             cosine_sim_type, sigmoid_weight, num_thresholds))
    sim_with_neg = tf_utils.to_sim_score(cosines, sim_type=cosine_sim_type, sigmoid_weight=sigmoid_weight)
    predictions = tf.reshape(sim_with_neg, [-1, 1])
    labels = tf.reshape(labels, [-1, 1])
    auc_value, auc_update_op = streaming_auc(predictions, labels, num_thresholds=num_thresholds)
    return auc_value, auc_update_op
