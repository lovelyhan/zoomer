#!/bin/env python
# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pai
import xdl

from xdl.utils import tfprint
from model.ops import optimizers
from model.ops import euler_ops
from xdl.utils import xdl2_dataio

from model import zoomer as Model
from model import zoomer_model_config as zoomer_model_config

option_task_test_auc = zoomer_model_config.option_task_test_auc
option_task_trace_ad = zoomer_model_config.option_task_trace_ad
option_task_traceM_query = zoomer_model_config.option_task_traceM_query
option_task_traceM_user = zoomer_model_config.option_task_traceM_user
option_task_model_clip = zoomer_model_config.option_task_model_clip
option_task_model_clip_test_auc = zoomer_model_config.option_task_model_clip_test_auc
option_task_model_clip_trace_qu = zoomer_model_config.option_task_model_clip_trace_qu
option_input_sample_id = zoomer_model_config.option_input_sample_id
option_input_unseen_flag = zoomer_model_config.option_input_unseen_flag
option_input_labels = zoomer_model_config.option_input_labels
option_input_query = zoomer_model_config.option_input_query
option_input_user = zoomer_model_config.option_input_user
option_input_ad = zoomer_model_config.option_input_ad
# option_input_ad_neg = zoomer_model_config.option_input_ad_neg
option_input_context = zoomer_model_config.option_input_context
option_input_request_ad_embedding = zoomer_model_config.option_input_request_ad_embedding
option_use_graph_data = zoomer_model_config.option_use_graph_data
option_use_data_prefetch = zoomer_model_config.option_use_data_prefetch
option_use_aggregation_embedding = zoomer_model_config.option_use_aggregation_embedding
option_use_request_aggregation = zoomer_model_config.option_use_request_aggregation
option_use_request_qu = zoomer_model_config.option_use_request_qu
option_use_request_ad = zoomer_model_config.option_use_request_ad
option_use_request_score = zoomer_model_config.option_use_request_score
option_use_feature_level_attention = zoomer_model_config.option_use_feature_level_attention
option_use_semantic_attention = zoomer_model_config.option_use_semantic_attention



def run(ctx):
    sparse_varlen_features_list = [
        {"feature_name": "query_id", "table_name": "query_id", "is_need_hash": False, "value_only": True},
        {"feature_name": "user_id",  "table_name": "user_id", "is_need_hash": False, "value_only": True},
        {"feature_name": "item_id",  "table_name": "item_id", "is_need_hash": False, "value_only": True},
        {"feature_name": "preitem",  "table_name": "item_id", "is_need_hash": False, "value_only": False},
        {"feature_name": "preshop",  "table_name": "shop_id", "is_need_hash": False, "value_only": False},
        {"feature_name": "prebrand", "table_name": "brand_id", "is_need_hash": False, "value_only": False},
        {"feature_name": "precate",  "table_name": "cate_id", "is_need_hash": False, "value_only": False},
        {"feature_name": "prerootcate", "table_name": "cate_id", "is_need_hash": False, "value_only": False},
    ]

    dense_fixedlen_features_list = [
        {"feature_name": "label", "dense_dim": 1, "dense_default": [0.], "dense_dtype": tf.float32},
    ]


    batch, hooks = xdl2_dataio.get_batch(ctx, sparse_varlen_features_list=sparse_varlen_features_list,
                                         dense_fixedlen_features_list=dense_fixedlen_features_list,
                                         nonsparse_varlen_features_list=nonsparse_varlen_features_list,
                                         )
    tfprint.simple_print_all("batch keys", batch.keys(), highlight=True)

    input_sparse_features_table_name_dict = {}
    for sparse_feature in sparse_varlen_features_list:
        input_sparse_features_table_name_dict[sparse_feature["feature_name"]] = sparse_feature["table_name"]
    model_input_config = {
        "input_sparse_features_table_name_dict": input_sparse_features_table_name_dict # 必须传进模型中
    }

    model_option = {
        option_task_test_auc : False,
        option_task_trace_ad : False,
        option_task_traceM_query : False,
        option_task_traceM_user : False,
        option_task_model_clip : False,
        option_task_model_clip_test_auc : False,
        option_task_model_clip_trace_qu: False,
        option_input_sample_id : False,
        option_input_unseen_flag : False,
        option_input_labels : True,
        option_input_query : True,
        option_input_user : True,
        option_input_ad : True,
        # option_input_ad_neg : True,
        option_input_context : True,
        option_input_request_ad_embedding : False,
        option_use_graph_data : True,
        option_use_data_prefetch : True,
        option_use_aggregation_embedding : False, #
        option_use_request_aggregation : True,
        option_use_request_qu : True,
        option_use_request_ad : True,
        option_use_request_score : True,
        option_use_feature_level_attention : True,
        option_use_semantic_attention : True,
    }
    zoomer_model = Model.ZoomerBaseModel(context=ctx, model_config=model_input_config)
    labels, cosine_ensemble_dict = zoomer_model.inference_with_option(batch, model_option, True)
    loss = zoomer_model.loss(labels, cosine_ensemble_dict)
    auc, auc_update_ops = zoomer_model.auc_train(labels, cosine_ensemble_dict)


    grads = tf.gradients(loss, tf.trainable_variables())
    for grad, var in list(zip(grads, tf.trainable_variables())):
        if grad is not None and isinstance(grad, tf.Tensor):
            grad = tf.reduce_sum(grad)
            grad = tfprint.tf_print_dense(grad, "gradient_sum---{}".format(var.name), first_n=10, summarize=32)
            loss = loss+grad*0.
    #######################################

    # tf.train.AdamOptimizer(0.1)
    # tf.train.AdagradOptimizer(0.1)
    optimizer, sync_replicas_hook = optimizers.get_optimizers(ctx, ctx.get("optimizers"))
    if sync_replicas_hook != None:
        hooks.append(sync_replicas_hook)

    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

    # run_ops = [loss, auc_update_op, train_op]
    run_ops = [train_op] + auc_update_ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    tfprint.simple_print_all("all variables", [v.name for v in tf.global_variables()], highlight=True)

    hooks.extend(zoomer_model.get_prefetch_hooks())
    if ctx.is_chief():
        hooks.extend(zoomer_model.get_hts_hooks())

    if ctx.is_chief():
        hooks.append(tf.train.ProfilerHook(ctx.get("profiler", "save_interval_steps"), output_dir=ctx.get("profiler", "output_dir")))

    tf.summary.scalar('auc', auc)
    tf.summary.scalar('loss', loss)
    hooks.append(
        tf.train.SummarySaverHook(
            save_steps=100000,
            output_dir=ctx.get("summary", "output_dir"),
            summary_op=tf.summary.merge_all()
        )
    )

    trainer = xdl.stage.Train(run_ops, hooks, log_step=ctx.get("log_step"), save_step=ctx.get("checkpoint", "save_interval_steps"))
    trainer.add_metrics('loss', loss)
    trainer.add_metrics('auc', auc)
    for name, value in zoomer_model.get_metrics():
        trainer.add_metrics(name, value)


    def write_auc(sess, ctx):
        tfprint.simple_print_all("ctx.stage", "write_auc", highlight=True, seg=":", data_time=True)
        if ctx.is_chief():
            auc_value = sess.run(auc)
            # content = ctx.get("auc", "auc_type") + ":" + str(auc_value)
            content = "train_auc:" + str(auc_value)
            tfprint.simple_print_all("Final Auc", auc_value, highlight=True, seg=":", data_time=True)
            tf.gfile.MakeDirs(ctx.get("auc", "auc_score_output_dir"))
            auc_path = ctx.get("auc", "auc_score_output_dir") + "/" + ctx.get("auc", "auc_checkpoint_model")
            with tf.gfile.Open(auc_path, mode="w") as f:
                f.write(content)

            # auc_threshold = 0.6
            # auc_threshold = ctx.get_default(None, "auc", "auc_threshold")
            # if auc_value is not None and auc_threshold is not None and auc_value < auc_threshold:
            #     err_info = "auc_value={} < auc_threshold({})".format(auc_value, auc_threshold)
            #     tfprint.simple_print_all("AUC check error", err_info, highlight=True, seg=":", data_time=True)
            #     raise ValueError("zoomer auc check error: {}".format(err_info))

    def train():
        ctx.stage('train', trainer())
        ctx.save()  # save checkpoint
        ctx.stage("write_auc", write_auc)
        if ctx.is_chief():
            tfprint.print_ckpt_all_vars(ctx.get('checkpoint', 'output_dir'))

    print("init_graph")
    euler_ops.init_graph(ctx)
    print("init_graph finished")
    ctx.start(train)




if __name__ == '__main__':
    cfg = tf.ConfigProto(tensor_fuse=True)
    cfg.gpu_options.allow_growth = True
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.1
    ctx = xdl.simple_context(session_config=cfg)

    tfprint.simple_print("xdl2.runner", __file__)
    tfprint.simple_print_all("run config", ctx.get(), highlight=True)

    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    with ctx.scope():
        run(ctx)