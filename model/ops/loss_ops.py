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



def focal_neg_log_loss_with_labels(predict_prob, labels, focal_weight = 2.0):
    """
    focal cross_entropy loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predict_prob)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predict_prob)
    loss = y * -log(x) + (1 - y) * -log(1 - x)
    """
    neg_log_loss = labels * (-tf.log(predict_prob)) + (1 - labels) * (-tf.log(1 - predict_prob))
    predictions_pt = tf.where(tf.equal(labels, 1), predict_prob, 1 - predict_prob)
    focal = tf.pow(1 - predictions_pt, focal_weight)
    focal_loss = focal * neg_log_loss
    focal_loss = tf.reduce_sum(focal_loss, axis=1)
    return focal_loss

def cosine_focal_cross_entropy_loss_with_labels(cosines, labels,
                                                cosine_sim_type="sigmoid", sigmoid_weight=1.0,
                                                focal_weight=2.0):
    tfprint.simple_print("cosine_focal_cross_entropy_loss_with_labels",
                         "cosine_sim_type={},sigmoid_weight={},focal_weight={}".format(
                             cosine_sim_type, sigmoid_weight, focal_weight))

    sims = tf_utils.to_sim_score(cosines, sim_type=cosine_sim_type, sigmoid_weight=sigmoid_weight)
    my_loss = focal_neg_log_loss_with_labels(sims, labels, focal_weight = focal_weight)
    return my_loss
