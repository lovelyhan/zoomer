# coding=utf-8

import tensorflow as tf
import pai
import xdl

from xdl.utils import tfprint
from xdl.utils import xdl2_embeddings, xdl2_trace_hashtable
from xdl.utils import tf_utils
from model.ops import euler_ops
from model.ops import tf_ops
from model.ops import loss_ops
from model.ops import auc_ops
from model.layers import dnn_layers
from model.layers import model_layers

from model import zoomer_model_config as zoomer_model_config


class ZoomerBaseModel(object):
    def __init__(self, context=None, model_name='ZoomerBaseModel', model_config=None):

        self.context = context
        self.name = model_name
        self.model_config = self.build_config(new_config=model_config)
        # self.common_config = zoomer_model_config

        # sparse_embedding
        # self.hts_handler = None
        self.hts_handler = xdl2_embeddings.XDL2SparseEmbedding(self.context)
        # layers
        self.layers_dict = {}  # {scope_name: {layer_name: LAYER}}
        # session run metrics
        self.metrics_kv_list = []
        self.trace_variables_kv_list = []
        self.signature_inputs = {}
        self.signature_outputs = {}

        # other config
        self.use_data_prefetch = self.context.get("reader", "use_data_prefetch")
        self.sigmoid_weight = 4.0  # 3.0, 4.0, 5.0
        self.cosine_sim_type = "sigmoid"  # None, sigmoid

    def build_config(self, new_config=None):
        # zoomer_para_list = dir(zoomer_model_config) # ['__builtins__', '__doc__', '__file__', '__name__', '__package__', ...]
        model_config = {}
        for attr in dir(zoomer_model_config):
            if not attr.startswith("__"):
                model_config[attr] = getattr(zoomer_model_config, attr)
        if new_config is not None:
            for k, v in new_config.items():
                model_config[k] = v


        node_used_sparse_feature_set = set(model_config["q_full_sparse_feature_names"] + \
                                           model_config["i_full_sparse_feature_names"] + \
                                           model_config["u_full_sparse_feature_names"] + \
                                           model_config["q_part_sparse_feature_names"] + \
                                           model_config["i_part_sparse_feature_names"]
                                           )
        # set(['query_id', 'q_terms'...])

        model_config["node_used_sparse_feature_set"] = node_used_sparse_feature_set
        used_sparse_features_table_name_dict = {} # {feature_name: table_name} 不仅包括node的，也包括input feature的
        for feature_name in node_used_sparse_feature_set:
            table_name = model_config["node_all_sparse_features_dict"][feature_name]["table_name"]
            if feature_name in used_sparse_features_table_name_dict:
                if used_sparse_features_table_name_dict[feature_name] != table_name:
                    raise ValueError("build used_sparse_features_table_name_dict ERROR1: "
                                     "feature_name={} has conflicted table_name ({}, {})".format(
                            feature_name, table_name, used_sparse_features_table_name_dict[feature_name]))
            else:
                used_sparse_features_table_name_dict[feature_name] = table_name
        for feature_name in model_config["input_sparse_features_table_name_dict"]:
            table_name = model_config["input_sparse_features_table_name_dict"][feature_name]
            if feature_name in used_sparse_features_table_name_dict:
                if used_sparse_features_table_name_dict[feature_name] != table_name:
                    raise ValueError("build used_sparse_features_table_name_dict ERROR2: "
                                     "feature_name={} has conflicted table_name ({}, {})".format(
                            feature_name, table_name, used_sparse_features_table_name_dict[feature_name]))
            else:
                used_sparse_features_table_name_dict[feature_name] = table_name
        model_config["used_sparse_features_table_name_dict"] = used_sparse_features_table_name_dict

        tfprint.simple_print_all("model_config", model_config, highlight=True)

        return model_config

    def get_hts_hooks(self):
        hooks = []
        if self.hts_handler != None and self.hts_handler.heuristic_strategy == True:
            hooks.append(xdl2_embeddings.get_partitioner_hook(self.context))
        return hooks
    def get_prefetch_hooks(self):
        hooks = []
        if self.use_data_prefetch:
            hooks.append(pai.data.make_prefetch_hook())
        return hooks
    def add_metrics(self, name, value):
        self.metrics_kv_list.append([name, value])
    def get_metrics(self):
        return self.metrics_kv_list
    def add_trace_variables(self, name, value):
        self.trace_variables_kv_list.append([name, value])
    def get_trace_variables(self):
        return self.trace_variables_kv_list

    def get_para(self, para_name):
        if para_name in self.model_config:
            return self.model_config[para_name]
        return getattr(zoomer_model_config, para_name)

    def get_node_feature_table_name(self, feature_name):
        return self.model_config["node_all_sparse_features_dict"][feature_name]["table_name"]

    def get_input_feature_table_name(self, feature_name):
        return self.model_config["input_sparse_features_table_name_dict"][feature_name]

    def get_table_config(self, table_name):
        return self.model_config["all_sparse_embed_hts_configs_dict"][table_name]

    def get_tables_config_dict(self, table_names):
        tables_config = {}
        for name in table_names:
            if name not in tables_config:
                tables_config[name] = self.model_config["all_sparse_embed_hts_configs_dict"][name]
        return tables_config
    
    def get_feature_table_name(self, feature_name):
        return self.model_config["used_sparse_features_table_name_dict"][feature_name]
    
    def get_feature_embed_dim(self, feature_name):
        table_name = self.get_feature_table_name(feature_name)
        return self.get_table_config(table_name)["dim"]

    def get_features_embed_dim_list(self, feature_names):
        dim_list = []
        for name in feature_names:
            dim_list.append(self.get_feature_embed_dim(name))
        return dim_list

    def get_features_embedding_list(self, feature_names, features_embed_dict):#不仅查dim
        embed_list = []
        for name in feature_names:
            embed_list.append(features_embed_dict[name])
        return embed_list

    def get_nbs_features_embedding_list_list(self, nb_type_names, nb_config_dict, nbs_features_embed_dict):
        nbs_features_embed_list_list = []
        for type_name in nb_type_names:
            # "nb_type_name": [nb_cnt, nb_edge_types, nb_feature_names]
            nb_embeds = self.get_features_embedding_list(nb_config_dict[type_name][2], nbs_features_embed_dict[type_name])
            nbs_features_embed_list_list.append(nb_embeds)
        return nbs_features_embed_list_list

    def get_sparse_feature_embedding(self, feature_name, feature_tensor):
        table_name = self.get_feature_table_name(feature_name)
        return self.hts_handler.get_sparse_embedding(table_name, feature_tensor)

    def get_sparse_feature_embedding_dict(self, features_dict):
        features_embedding_dict = {}
        for name, tensors_or_dict in features_dict.items():
            if isinstance(tensors_or_dict, dict):
                embed = self.get_sparse_feature_embedding_dict(tensors_or_dict)
            else:
                embed = self.get_sparse_feature_embedding(name, tensors_or_dict)
            features_embedding_dict[name] = embed
        return features_embedding_dict

    def get_sparse_feature_embedding_dict_all(self, features_dict_list):
        ret = []
        for features_dict in features_dict_list:
            ret.append(self.get_sparse_feature_embedding_dict(features_dict))
        return ret
    
    def _get_request_score(self, req_src_embedding, req_dst_embedding, score_type="cosine"):
        if score_type == "cosine":
            score = tf_ops.cosine_fun(req_src_embedding, req_dst_embedding)
        elif score_type == "inner_product":
            score = tf_ops.get_inner_product(req_src_embedding, req_dst_embedding)
        else:
            raise ValueError("score_type = {} is not supported!".format(score_type))
        return score

    def get_request_score_or_list(self, req_src_embedding, req_dst_embeddings, dst_num=1, embed_dim=None,
                                  score_type="cosine"):
        if dst_num == 1:
            return self._get_request_score(req_src_embedding, req_dst_embeddings, score_type=score_type)
        # batch_size = tf.shape(req_src_embedding)[0]
        # embed_dim = tf.shape(req_src_embedding)[1]
        score_list = []
        req_dst_embeddings_reshape = tf.reshape(req_dst_embeddings, [-1, dst_num, embed_dim])
        for i in range(dst_num):
            dst_i_vec = tf.reshape(req_dst_embeddings_reshape[:, i, :], [-1, embed_dim])
            score_i = self._get_request_score(req_src_embedding, dst_i_vec, score_type=score_type)
            score_list.append(score_i)
        return score_list

    def create_feature_aggregation_layer(self, scope_name, layer_name, feature_names, input_dim,
                                         mid_dim=None, output_dim=None, init_type=None, init_para=None):
        assert layer_name not in self.layers_dict[scope_name], \
            "create_feature_aggregation_layer ERROR!!! layer_name={} already exists!".format(layer_name)
        features_dim_list = self.get_features_embed_dim_list(feature_names)
        _layer = FeatureAggregationLayer(layer_name=layer_name,
                                         features_dim_list=features_dim_list,
                                         input_dim=input_dim,
                                         mid_dim=mid_dim, output_dim=output_dim,
                                         init_type=init_type, init_para=init_para)
        self.layers_dict[scope_name][layer_name] = _layer


    def create_request_aggregation_layer(self, scope_name, layer_name, sub_layer_num, input_dim=None,
                                         mid_dim=None, output_dim=None, init_type=None, init_para=None):
        assert layer_name not in self.layers_dict[scope_name], \
            "create_request_aggregation_layer ERROR!!! layer_name={} already exists!".format(layer_name)
        _layer = model_layers.DSSMLayer(layer_name=layer_name, sub_layer_num=sub_layer_num,
                                        input_dim=input_dim, mid_dim=mid_dim, output_dim=output_dim,
                                        init_type=init_type, init_para=init_para)
        self.layers_dict[scope_name][layer_name] = _layer


    def _build_graph_embedding(self, scope_name):   # 构造embedding的hash table
        assert scope_name not in self.layers_dict, \
            "_build_graph_embedding ERROR!!! scope_name={} already exists!".format(scope_name)
        table_names_list = [self.get_node_feature_table_name(v) for v in self.model_config["node_used_sparse_feature_set"]]

        tables_config = self.get_tables_config_dict(table_names_list)
        tfprint.simple_print_all("_build_graph_embedding", tables_config, highlight=True)
        # if self.hts_handler == None:
        #     self.hts_handler = xdl2_embeddings.XDL2SparseEmbedding(self.context, tables_config = tables_config)
        self.hts_handler.create_hashtable_from_config(tables_config)

    def _build_graph_aggregation_embedding(self, scope_name, table_names_list=None):
        assert scope_name not in self.layers_dict, \
            "_build_graph_aggregation_embedding ERROR!!! scope_name={} already exists!".format(scope_name)
        # self.layers_dict[scope_name] = {}
        if table_names_list == None:
            return
        # tables_config = self.get_tables_config_dict(table_names_list)
        tables_config = {}
        for name in table_names_list:
            if name not in tables_config:
                tables_config[name] = self.model_config["all_aggregation_embed_hts_configs_dict"][name]
        tfprint.simple_print_all("_build_graph_aggregation_embedding", tables_config, highlight=True)
        self.hts_handler.create_hashtable_from_config(tables_config)
    
    def _build_context_embedding(self, scope_name):
        assert scope_name not in self.layers_dict, \
            "_build_context_embedding ERROR!!! scope_name={} already exists!".format(scope_name)
        # self.layers_dict[scope_name] = {}
        urb_sparse_feature_names = self.model_config["urb_sparse_feature_names"]
        table_names_list = [self.get_feature_table_name(v) for v in urb_sparse_feature_names]
        tables_config = self.get_tables_config_dict(table_names_list)
        tfprint.simple_print_all("_build_context_embedding", tables_config, highlight=True)
        self.hts_handler.create_hashtable_from_config(tables_config)
    
    def _build_graph_feature_aggregation(self, scope_name):
        """ features embedding -> node embedding"""
        assert scope_name not in self.layers_dict, \
            "_build_graph_feature_aggregation ERROR!!! scope_name={} already exists!".format(scope_name)
        self.layers_dict[scope_name] = {}

        dense_init_type = self.model_config["dense_init_type"]
        dense_init_para = self.model_config["dense_init_para"]
        node_embed_dim = self.model_config["node_embed_dim"]
        mapped_out_dim = self.model_config["mapped_out_dim"]
        q_full_sparse_feature_names = self.model_config["q_full_sparse_feature_names"]
        i_full_sparse_feature_names = self.model_config["i_full_sparse_feature_names"]
        u_full_sparse_feature_names = self.model_config["u_full_sparse_feature_names"]

        tfprint.simple_print_all("_build_graph_feature_aggregation", "q_node, i_node, u_node", highlight=True)

        self.create_feature_aggregation_layer(scope_name,
                                              layer_name="q_node",
                                              feature_names=q_full_sparse_feature_names, input_dim=mapped_out_dim,
                                              mid_dim=node_embed_dim*2, output_dim=node_embed_dim,
                                              init_type=dense_init_type, init_para=dense_init_para)
        self.create_feature_aggregation_layer(scope_name,
                                              layer_name="i_node",
                                              feature_names=i_full_sparse_feature_names, input_dim=mapped_out_dim,
                                              mid_dim=node_embed_dim*2, output_dim=node_embed_dim,
                                              init_type=dense_init_type, init_para=dense_init_para)
        self.create_feature_aggregation_layer(scope_name,
                                              layer_name="u_node",
                                              feature_names=u_full_sparse_feature_names, input_dim=mapped_out_dim,
                                              mid_dim=node_embed_dim*2, output_dim=node_embed_dim,
                                              init_type=dense_init_type, init_para=dense_init_para)
    
    
    def _build_request_aggregation(self, scope_name, request_qu=True, request_ad=True):
        assert scope_name not in self.layers_dict, \
            "_build_request_aggregation ERROR!!! scope_name={} already exists!".format(scope_name)
        self.layers_dict[scope_name] = {}

        dense_init_type = self.model_config["dense_init_type"]
        dense_init_para = self.model_config["dense_init_para"]
        node_embed_dim = self.model_config["node_embed_dim"]
        nbs_agg_embed_dim = self.model_config["nbs_agg_embed_dim"]
        node_agg_embed_dim = self.model_config["node_agg_embed_dim"]
        # urb_embed_dim = self.model_config["urb_embed_dim"]
        req_embed_dim = self.model_config["req_embed_dim"]
        input_dim = self.model_config["nbs_agg_embed_dim"]

        sub_layer_num = 2 # 3

        if request_qu:
            tfprint.simple_print_all("_build_request_aggregation", "qu_self", highlight=True)
            self.create_request_aggregation_layer(scope_name,
                                              layer_name="qu_self", sub_layer_num=sub_layer_num,
                                              input_dim=input_dim * 2,
                                              mid_dim=input_dim, output_dim=input_dim,
                                              init_type=dense_init_type, init_para=dense_init_para)
        if request_ad:
            tfprint.simple_print_all("_build_request_aggregation", "ad_self", highlight=True)
            self.create_request_aggregation_layer(scope_name,
                                              layer_name="ad_self", sub_layer_num=sub_layer_num,
                                              input_dim=input_dim,
                                              mid_dim=input_dim, output_dim=input_dim,
                                              init_type=dense_init_type, init_para=dense_init_para)
        # self.create_request_aggregation_layer(scope_name,
        #                                       layer_name="qu_ad_share", sub_layer_num=sub_layer_num,
        #                                       input_dim=req_embed_dim,
        #                                       mid_dim=req_embed_dim, output_dim=req_embed_dim,
        #                                       init_type=dense_init_type, init_para=dense_init_para)


    def add_context_feature(self, node_features_emb, aux_node_features_emb, nbs_count):
        aux_node_features_emb = tf.expand_dims(aux_node_features_emb, axis=-2)
        node_features_emb_dim = node_features_emb.shape[-1].value

        node_features_emb = tf.reshape(node_features_emb, [-1, nbs_count, node_features_emb_dim])
        context_emb = tf.add(aux_node_features_emb, node_features_emb)
        context_emb = tf.reshape(context_emb, [-1, node_features_emb_dim])

        return context_emb

    def create_feature_mapping_layer(self, scope_name, layer_name, input_dim, output_dim, init_type, init_para):
        _layer = dnn_layers.DnnSingleLayer(layer_name=layer_name, input_dim=input_dim, output_dim=output_dim,
                                                weight_init_type=init_type, weight_init_para=init_para,
                                                has_biases = True, act_name="")
        self.layers_dict[scope_name][layer_name] = _layer


    def _build_feature_mapping(self, scope_name):
        self.layers_dict[scope_name] = {}
        dense_init_type = self.model_config["dense_init_type"]
        dense_init_para = self.model_config["dense_init_para"]
        mapped_out_dim = self.model_config["mapped_out_dim"]
        q_input_dim = sum(self.get_features_embed_dim_list(self.model_config["q_full_sparse_feature_names"]))
        u_input_dim = sum(self.get_features_embed_dim_list(self.model_config["u_full_sparse_feature_names"]))
        ad_input_dim = sum(self.get_features_embed_dim_list(self.model_config["i_full_sparse_feature_names"]))
        assert q_input_dim == 80, \
            "q_input_dim ERROR!!! q_input_dim={} ".format(q_input_dim)
        assert u_input_dim == 176, \
            "u_input_dim ERROR!!! u_input_dim={} ".format(u_input_dim)
        assert ad_input_dim == 160, \
            "ad_input_dim ERROR!!! ad_input_dim={} ".format(ad_input_dim)
        self.create_feature_mapping_layer(scope_name,
                                                layer_name="q_mapping",
                                                input_dim=q_input_dim, output_dim=mapped_out_dim,
                                                init_type=dense_init_type, init_para=dense_init_para)
        self.create_feature_mapping_layer(scope_name,
                                                layer_name="u_mapping",
                                                input_dim=u_input_dim, output_dim=mapped_out_dim,
                                                init_type=dense_init_type, init_para=dense_init_para)
        self.create_feature_mapping_layer(scope_name,
                                                layer_name="i_mapping",
                                                input_dim=ad_input_dim, output_dim=mapped_out_dim,
                                                init_type=dense_init_type, init_para=dense_init_para)

    def create_feature_nbs_aggregation_layer(self, scope_name, layer_name, node_feature_names,
                                             nb_type_names, nb_config_dict,
                                             mid_dim=None, output_dim=None, option_use_semantic_attention=None,
                                             init_type=None, init_para=None):
        assert layer_name not in self.layers_dict[scope_name], \
            "create_feature_nbs_aggregation_layer ERROR!!! layer_name={} already exists!".format(layer_name)
        node_features_dim_list = self.get_features_embed_dim_list(node_feature_names)
        nbs_count_list = []
        nbs_features_dim_list_list = []
        for type_name in nb_type_names:
            # "nb_type_name": [nb_cnt, nb_edge_types, nb_feature_names]
            nb_cnt = nb_config_dict[type_name][0]
            nb_feature_names = nb_config_dict[type_name][2]
            nbs_count_list.append(nb_cnt)
            nbs_features_dim_list_list.append(self.get_features_embed_dim_list(nb_feature_names))
        _layer = FeatureNbsAggregationLayer(layer_name=layer_name,
                                            node_features_dim_list=node_features_dim_list,
                                            nbs_features_dim_list_list=nbs_features_dim_list_list,
                                            nbs_count_list=nbs_count_list,
                                            mid_dim=mid_dim, output_dim=output_dim, 
                                            option_use_semantic_attention = option_use_semantic_attention,
                                            init_type=init_type, init_para=init_para)
        self.layers_dict[scope_name][layer_name] = _layer

    def _build_graph_feature_nbs_aggregation(self, scope_name, option_use_semantic_attention):
        assert scope_name not in self.layers_dict, \
            "_build_graph_feature_nbs_aggregation ERROR!!! scope_name={} already exists!".format(scope_name)
        self.layers_dict[scope_name] = {}

        dense_init_type = self.model_config["dense_init_type"]
        dense_init_para = self.model_config["dense_init_para"]
        nbs_agg_embed_dim = self.model_config["nbs_agg_embed_dim"]
        q_full_sparse_feature_names = self.model_config["q_full_sparse_feature_names"]
        i_full_sparse_feature_names = self.model_config["i_full_sparse_feature_names"]
        u_full_sparse_feature_names = self.model_config["u_full_sparse_feature_names"]
        q_nb_config_dict = self.model_config["q_nb_config_dict"]
        i_nb_config_dict = self.model_config["i_nb_config_dict"]
        u_nb_config_dict = self.model_config["u_nb_config_dict"]
        q_nb_type_names = self.model_config["q_nb_type_names"]
        i_nb_type_names = self.model_config["i_nb_type_names"]
        u_nb_type_names = self.model_config["u_nb_type_names"]

        tfprint.simple_print_all("_build_graph_feature_nbs_aggregation", "q_nbs, i_nbs, u_nbs", highlight=True)

        self.create_feature_nbs_aggregation_layer(scope_name,
                                                  layer_name="q_nbs",
                                                  node_feature_names=q_full_sparse_feature_names,
                                                  nb_type_names=q_nb_type_names,
                                                  nb_config_dict=q_nb_config_dict,
                                                  mid_dim=nbs_agg_embed_dim * 2, output_dim=nbs_agg_embed_dim,
                                                  option_use_semantic_attention=option_use_semantic_attention,
                                                  init_type=dense_init_type, init_para=dense_init_para)
        self.create_feature_nbs_aggregation_layer(scope_name,
                                                  layer_name="i_nbs",
                                                  node_feature_names=i_full_sparse_feature_names,
                                                  nb_type_names=i_nb_type_names,
                                                  nb_config_dict=i_nb_config_dict,
                                                  mid_dim=nbs_agg_embed_dim * 2, output_dim=nbs_agg_embed_dim,
                                                  option_use_semantic_attention=option_use_semantic_attention,
                                                  init_type=dense_init_type, init_para=dense_init_para)
        self.create_feature_nbs_aggregation_layer(scope_name,
                                                  layer_name="u_nbs",
                                                  node_feature_names=u_full_sparse_feature_names,
                                                  nb_type_names=u_nb_type_names,
                                                  nb_config_dict=u_nb_config_dict,
                                                  mid_dim=nbs_agg_embed_dim * 2, output_dim=nbs_agg_embed_dim,
                                                  option_use_semantic_attention=option_use_semantic_attention,
                                                  init_type=dense_init_type, init_para=dense_init_para)
        for node_type in ["q", "i", "u"]:
            self.create_feature_nbs_aggregation_layer(scope_name,
                                                    layer_name=node_type+"_q_nbs_2",
                                                    node_feature_names=q_full_sparse_feature_names,
                                                    nb_type_names=q_nb_type_names,
                                                    nb_config_dict=q_nb_config_dict,
                                                    mid_dim=nbs_agg_embed_dim * 2, output_dim=nbs_agg_embed_dim*2,
                                                    option_use_semantic_attention=option_use_semantic_attention,
                                                    init_type=dense_init_type, init_para=dense_init_para)
            self.create_feature_nbs_aggregation_layer(scope_name,
                                                    layer_name=node_type+"_i_nbs_2",
                                                    node_feature_names=i_full_sparse_feature_names,
                                                    nb_type_names=i_nb_type_names,
                                                    nb_config_dict=i_nb_config_dict,
                                                    mid_dim=nbs_agg_embed_dim * 2, output_dim=nbs_agg_embed_dim*2,
                                                    option_use_semantic_attention=option_use_semantic_attention,
                                                    init_type=dense_init_type, init_para=dense_init_para)
            # self.create_feature_nbs_aggregation_layer(scope_name,
            #                                         layer_name=node_type+"_u_nbs_2",
            #                                         node_feature_names=u_full_sparse_feature_names,
            #                                         nb_type_names=u_nb_type_names,
            #                                         nb_config_dict=u_nb_config_dict,
            #                                         mid_dim=nbs_agg_embed_dim * 2, output_dim=nbs_agg_embed_dim*2,
            #                                         init_type=dense_init_type, init_para=dense_init_para)


    
    def dict_2_list(self, input_dict):
        ret = []
        for item in input_dict:
            ret.append(input_dict[item])
        return ret

    def _build_feature_level_attention_layer(self, scope_name):
        dense_init_type = self.model_config["dense_init_type"]
        dense_init_para = self.model_config["dense_init_para"]
        feature_dim_dict = self.model_config["feature_dim_dict"]
        _layer = FeatureLevelAttention(layer_name='FeatureLevelAttentionLayer', feature_dim_dict=feature_dim_dict,
                                       init_type=dense_init_type, init_para=dense_init_para)
        self.layers_dict[scope_name] = _layer

    def inference_with_option(self, batch, model_option, training):
        option_task_trace_tmp = zoomer_model_config.option_task_trace_tmp
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

        pv_ad_num = self.model_config["neg_pv_num"]
        #neg_ad_num = self.model_config["neg_num"]
        #neg_feat_num = self.model_config["neg_num"]

        node_embed_dim = self.model_config["node_embed_dim"]
        nbs_agg_embed_dim = self.model_config["nbs_agg_embed_dim"]
        node_agg_embed_dim = self.model_config["node_agg_embed_dim"]
        # urb_embed_dim = self.model_config["urb_embed_dim"]
        req_embed_dim = self.model_config["req_embed_dim"]
        space_mapping_out_dim = self.model_config["space_mapping_out_dim"]


        q_full_sparse_feature_names = self.model_config["q_full_sparse_feature_names"]
        i_full_sparse_feature_names = self.model_config["i_full_sparse_feature_names"]
        u_full_sparse_feature_names = self.model_config["u_full_sparse_feature_names"]
        q_nb_config_dict = self.model_config["q_nb_config_dict"]
        i_nb_config_dict = self.model_config["i_nb_config_dict"]
        u_nb_config_dict = self.model_config["u_nb_config_dict"]
        q_nb_type_names = self.model_config["q_nb_type_names"]
        i_nb_type_names = self.model_config["i_nb_type_names"]
        u_nb_type_names = self.model_config["u_nb_type_names"]
        nbs_cnt = self.model_config["nb_cnt"]
        nbs_2_cnt = self.model_config["nb_2_cnt"]
        #urb_sparse_feature_names = self.model_config["urb_sparse_feature_names"]

        ####### batch input
        prefetch_data = {}
        if model_option[option_input_sample_id]:
            # sample_id = batch["sample_id"]  # [batch_size]
            prefetch_data["sample_id"] = batch["sample_id"]
        if model_option[option_input_unseen_flag]:
            # unseen_flag = batch["unseen_flag"]  # [batch_size]
            prefetch_data["unseen_flag"] = batch["unseen_flag"]
        if model_option[option_input_labels]:
            # labels = batch["label"]  # dense, [batch_size, 1]
            prefetch_data["labels"] = batch["label"]
        if model_option[option_input_query]:
            # q_nodes = batch["query_id"]  # dense, [batch_size]
            prefetch_data["q_nodes"] = batch["query_id"]
        if model_option[option_input_user]:
            # u_nodes = batch["user_id"]  # dense, [batch_size]
            prefetch_data["u_nodes"] = batch["user_id"]
        if model_option[option_input_ad]:
            # ad_nodes = batch["item_id"]  # dense, [batch_size]
            prefetch_data["ad_nodes"] = batch["item_id"]

        if model_option[option_input_request_ad_embedding]:
            prefetch_data["request_ad_embedding"] = batch["request_ad_embedding"]

        if option_task_trace_tmp in model_option and model_option[option_task_trace_tmp]:
            # prefetch_data["feature_id"] = batch["feature_id"]
            prefetch_data["query_id"] = batch["query_id"]
            prefetch_data["user_id"] = batch["user_id"]
            prefetch_data["item_id"] = batch["item_id"]
            prefetch_data["cate_id"] = batch["cate_id"]

        ###### graph data
        if model_option[option_input_query] and model_option[option_use_graph_data]:
            t_q_node_features_dict, t_q_nbs_features_dict, t_q_nbs_2_features_dict = euler_ops.assemble_graph_sparse(
                                                                prefetch_data["q_nodes"],
                                                                n_sparse_feature_names=q_full_sparse_feature_names,
                                                                nb_config_dict=self.model_config["q_nb_config_dict"])
            prefetch_data["q_node_features_dict"] = t_q_node_features_dict
            prefetch_data["q_nbs_features_dict"] = t_q_nbs_features_dict
            prefetch_data["q_nbs_2_features_dict"] = t_q_nbs_2_features_dict


        if model_option[option_input_user] and model_option[option_use_graph_data]:
            t_u_node_features_dict, t_u_nbs_features_dict, t_u_nbs_2_features_dict = euler_ops.assemble_graph_sparse(
                                                                prefetch_data["u_nodes"],
                                                                n_sparse_feature_names=u_full_sparse_feature_names,
                                                                nb_config_dict=self.model_config["u_nb_config_dict"])

            prefetch_data["u_node_features_dict"] = t_u_node_features_dict
            prefetch_data["u_nbs_features_dict"] = t_u_nbs_features_dict
            prefetch_data["u_nbs_2_features_dict"] = t_u_nbs_2_features_dict

        if model_option[option_input_ad] and model_option[option_use_graph_data]:
            t_ad_node_features_dict, t_ad_nbs_features_dict, t_ad_nbs_2_features_dict = euler_ops.assemble_graph_sparse(
                                                                prefetch_data["ad_nodes"],
                                                                n_sparse_feature_names=i_full_sparse_feature_names,
                                                                nb_config_dict=self.model_config["i_nb_config_dict"])
            prefetch_data["ad_node_features_dict"] = t_ad_node_features_dict
            prefetch_data["ad_nbs_features_dict"] = t_ad_nbs_features_dict
            prefetch_data["ad_nbs_2_features_dict"] = t_ad_nbs_2_features_dict
        
        ###### prefetch
        # self.use_data_prefetch = self.context.get("reader", "use_data_prefetch")
        if option_use_data_prefetch in model_option:
            self.use_data_prefetch = model_option[option_use_data_prefetch]
        tfprint.simple_print("data prefetch for euler graph", "use_data_prefetch = {}".format(self.use_data_prefetch))
        if self.use_data_prefetch:
            prefetch_data_after = pai.data.prefetch(prefetch_data,
                                                    num_threads=self.context.get("reader", "thread_num"),
                                                    timeout_millis=800000)
            prefetch_data = prefetch_data_after

    
        if model_option[option_task_test_auc] or model_option[option_task_model_clip_test_auc] or model_option[option_task_model_clip_trace_qu]:
            sample_id_str = tf.expand_dims(prefetch_data["sample_id"], -1)
            unseen_flag_str = tf.as_string(tf.expand_dims(prefetch_data["unseen_flag"], -1))
            self.add_trace_variables("sample_id", sample_id_str)
            self.add_trace_variables("unseen_flag", unseen_flag_str)
            self.add_trace_variables("labels", prefetch_data["labels"])
        
        if model_option[option_task_trace_ad]:
            sample_id_str = tf.expand_dims(prefetch_data["sample_id"], -1)
            ad_nodes_str = tf.as_string(tf.expand_dims(prefetch_data["ad_nodes"], -1))
            self.add_trace_variables("ad_item_id", sample_id_str)
            self.add_trace_variables("ad_item_id_hash64", ad_nodes_str)

        if option_task_trace_tmp in model_option and model_option[option_task_trace_tmp]:
            sample_id_str = tf.expand_dims(prefetch_data["sample_id"], -1)
            self.add_trace_variables("sample_id", sample_id_str)
            self.add_trace_variables("labels", prefetch_data["labels"])
            # feature_id_str = tf.as_string(tf.expand_dims(prefetch_data["feature_id"].values, -1))
            # self.add_trace_variables("feature_id", feature_id_str)


        this_scope_name = "embedding"
        with tf.variable_scope(this_scope_name, reuse=tf.AUTO_REUSE) as scope:
            if model_option[option_use_graph_data]:
                self._build_graph_embedding(this_scope_name)
                # self._build_context_embedding(this_scope_name)
            else:
                self._build_context_embedding(this_scope_name)

            # 获取 feature embedding
            if model_option[option_input_query] and model_option[option_use_graph_data]:
                q_node_features_embedding_dict = self.get_sparse_feature_embedding_dict(prefetch_data["q_node_features_dict"])
                q_nbs_features_embedding_dict = self.get_sparse_feature_embedding_dict(prefetch_data["q_nbs_features_dict"])
                q_nbs_2_features_embedding_dict = self.get_sparse_feature_embedding_dict(prefetch_data["q_nbs_2_features_dict"])
            if model_option[option_input_user] and model_option[option_use_graph_data]:
                u_node_features_embedding_dict = self.get_sparse_feature_embedding_dict(prefetch_data["u_node_features_dict"])
                u_nbs_features_embedding_dict = self.get_sparse_feature_embedding_dict(prefetch_data["u_nbs_features_dict"])
                u_nbs_2_features_embedding_dict = self.get_sparse_feature_embedding_dict(prefetch_data["u_nbs_2_features_dict"]) 
            if model_option[option_input_ad] and model_option[option_use_graph_data]:
                ad_node_features_embedding_dict = self.get_sparse_feature_embedding_dict(prefetch_data["ad_node_features_dict"])
                ad_nbs_features_embedding_dict = self.get_sparse_feature_embedding_dict(prefetch_data["ad_nbs_features_dict"])
                ad_nbs_2_features_embedding_dict = self.get_sparse_feature_embedding_dict(prefetch_data["ad_nbs_2_features_dict"])
            

        # for feature_name in q_nbs_2_features_embedding_dict:
        #     for feature_name_2 in q_nbs_2_features_embedding_dict[feature_name]:
        #         for feature_name_3 in q_nbs_2_features_embedding_dict[feature_name][feature_name_2]:
        #             q_nbs_2_features_embedding_dict[feature_name][feature_name_2][feature_name_3] = tfprint.tf_print_dense(q_nbs_2_features_embedding_dict[feature_name][feature_name_2][feature_name_3], 
        #                                                             "q_nbs_features_embedding_mapped_dict", first_n=20, summarize=128)
 

        ################### trace_tmp
        if option_task_trace_tmp in model_option and model_option[option_task_trace_tmp]:
            # tmp_feature_embedding = self.get_sparse_feature_embedding(feature_name = "feature_id",
            #                                                           feature_tensor = prefetch_data["feature_id"])
            # self.add_trace_variables("feature_embedding", tmp_feature_embedding)
            for feature_name in ["query_id", "user_id", "item_id", "cate_id"]:
                tmp_feature_embedding = self.get_sparse_feature_embedding(feature_name=feature_name,
                                                                      feature_tensor=prefetch_data[feature_name])
                self.add_trace_variables(feature_name+"_embedding", tmp_feature_embedding)
            return tmp_feature_embedding
        ###################

        
        if option_use_feature_level_attention:
            this_scope_name = "feature_level_attention"
            with tf.variable_scope(this_scope_name, reuse=tf.AUTO_REUSE) as scope:
                self._build_feature_level_attention_layer(this_scope_name)
                q_node_features_embedding_dict = self.layers_dict[this_scope_name].inference(q_node_features_embedding_dict, u_node_features_embedding_dict, dict_depth=0, nb_cnt=1)
                q_nbs_features_embedding_dict = self.layers_dict[this_scope_name].inference(q_nbs_features_embedding_dict, u_node_features_embedding_dict, dict_depth=1, nb_cnt=nbs_cnt)
                q_nbs_2_features_embedding_dict = self.layers_dict[this_scope_name].inference(q_nbs_2_features_embedding_dict, u_node_features_embedding_dict, dict_depth=2, nb_cnt=nbs_cnt*nbs_2_cnt)
                
                u_node_features_embedding_dict = self.layers_dict[this_scope_name].inference(u_node_features_embedding_dict, u_node_features_embedding_dict, dict_depth=0, nb_cnt=1)
                u_nbs_features_embedding_dict = self.layers_dict[this_scope_name].inference(u_nbs_features_embedding_dict, u_node_features_embedding_dict, dict_depth=1, nb_cnt=nbs_cnt)
                u_nbs_2_features_embedding_dict = self.layers_dict[this_scope_name].inference(u_nbs_2_features_embedding_dict, u_node_features_embedding_dict, dict_depth=2, nb_cnt=nbs_cnt*nbs_2_cnt)
                
                ad_node_features_embedding_dict = self.layers_dict[this_scope_name].inference(ad_node_features_embedding_dict, u_node_features_embedding_dict, dict_depth=0, nb_cnt=1)
                ad_nbs_features_embedding_dict = self.layers_dict[this_scope_name].inference(ad_nbs_features_embedding_dict, u_node_features_embedding_dict, dict_depth=1, nb_cnt=nbs_cnt)
                ad_nbs_2_features_embedding_dict = self.layers_dict[this_scope_name].inference(ad_nbs_2_features_embedding_dict, u_node_features_embedding_dict, dict_depth=2, nb_cnt=nbs_cnt*nbs_2_cnt)

        # Feature mappint layer, mapping the feature of different types to the same dimension 
        this_scope_name = "feature_mapping"
        with tf.variable_scope(this_scope_name, reuse=tf.AUTO_REUSE) as scope:
            self._build_feature_mapping(this_scope_name)
            features_embedding_list = self.get_features_embedding_list(self.model_config["q_full_sparse_feature_names"], q_node_features_embedding_dict)
            q_node_features_embedding_mapped = self.layers_dict[this_scope_name]["q_mapping"].inference(
                tf.concat(features_embedding_list, 1)
                )
            # q_node_features_embedding_mapped = tf.layers.batch_normalization(q_node_features_embedding_mapped, axis=-1)
            # q_node_features_embedding_mapped = tf_ops.get_normed_tensor(q_node_features_embedding_mapped, axis=1)
            q_nbs_features_embedding_mapped_dict = {}
            q_nbs_2_features_embedding_mapped_dict = {}
            for nb_type_name in self.model_config["q_nb_type_names"]:
                q_nbs_2_features_embedding_mapped_dict[nb_type_name] = {}
                mapping_layer_name = nb_type_name+'_mapping'
                nbs_feature_names = self.model_config[nb_type_name+'_part_sparse_feature_names']
                features_embedding_list = self.get_features_embedding_list(nbs_feature_names, q_nbs_features_embedding_dict[nb_type_name])
                features_embedding = tf.concat(features_embedding_list, 1)
                q_nbs_features_embedding_mapped_dict[nb_type_name] = self.layers_dict[this_scope_name][mapping_layer_name].inference(
                    features_embedding
                    )
                q_nbs_features_embedding_mapped_dict[nb_type_name] = tfprint.tf_print_dense(q_nbs_features_embedding_mapped_dict[nb_type_name], 
                                                                    "q_nbs_features_embedding_mapped_dict", first_n=20, summarize=128)
             
                q_nbs_features_embedding_mapped_dict[nb_type_name] = tfprint.tf_print_dense(q_nbs_features_embedding_mapped_dict[nb_type_name], 
                                                                    "q_nbs_features_embedding_mapped_dict_normed", first_n=20, summarize=128)

                for nb_2_type_name in self.model_config["nb_config_dict_dict"][nb_type_name]:
                    mapping_layer_name = nb_2_type_name+'_mapping'
                    nbs_2_feature_names = self.model_config[nb_2_type_name+'_part_sparse_feature_names']
                    features_embedding_list = self.get_features_embedding_list(nbs_2_feature_names, q_nbs_2_features_embedding_dict[nb_type_name][nb_2_type_name])
                    features_embedding = tf.concat(features_embedding_list, 1)
                    q_nbs_2_features_embedding_mapped_dict[nb_type_name][nb_2_type_name] = self.layers_dict[this_scope_name][mapping_layer_name].inference(
                        features_embedding
                        )
                 
            features_embedding_list = self.get_features_embedding_list(self.model_config["u_full_sparse_feature_names"], u_node_features_embedding_dict)
            u_node_features_embedding_mapped = self.layers_dict[this_scope_name]["u_mapping"].inference(
                tf.concat(features_embedding_list, 1)
                )
            u_nbs_features_embedding_mapped_dict = {}
            u_nbs_2_features_embedding_mapped_dict = {}
            for nb_type_name in self.model_config["u_nb_type_names"]:
                u_nbs_2_features_embedding_mapped_dict[nb_type_name] = {}
                mapping_layer_name = nb_type_name+'_mapping'
                nbs_feature_names = self.model_config[nb_type_name+'_part_sparse_feature_names']
                features_embedding_list = self.get_features_embedding_list(nbs_feature_names, u_nbs_features_embedding_dict[nb_type_name])
                u_nbs_features_embedding_mapped_dict[nb_type_name] = self.layers_dict[this_scope_name][mapping_layer_name].inference(
                    tf.concat(features_embedding_list, 1)
                    )
                for nb_2_type_name in self.model_config["nb_config_dict_dict"][nb_type_name]:
                    mapping_layer_name = nb_2_type_name+'_mapping'
                    nbs_2_feature_names = self.model_config[nb_2_type_name+'_part_sparse_feature_names']
                    features_embedding_list = self.get_features_embedding_list(nbs_2_feature_names, u_nbs_2_features_embedding_dict[nb_type_name][nb_2_type_name])
                    features_embedding = tf.concat(features_embedding_list, 1)
                    u_nbs_2_features_embedding_mapped_dict[nb_type_name][nb_2_type_name] = self.layers_dict[this_scope_name][mapping_layer_name].inference(
                        features_embedding
                        )

            
            features_embedding_list = self.get_features_embedding_list(self.model_config["i_full_sparse_feature_names"], ad_node_features_embedding_dict)
            ad_node_features_embedding_mapped = self.layers_dict[this_scope_name]["i_mapping"].inference(
                tf.concat(features_embedding_list, 1)
                )
            ad_node_features_embedding_mapped = tfprint.tf_print_dense(ad_node_features_embedding_mapped, "ad_node_features_embedding_mapped", first_n=20, summarize=256)
            ad_nbs_features_embedding_mapped_dict = {}
            ad_nbs_2_features_embedding_mapped_dict = {}
            for nb_type_name in self.model_config["i_nb_type_names"]:
                ad_nbs_2_features_embedding_mapped_dict[nb_type_name] = {}
                mapping_layer_name = nb_type_name+'_mapping'
                nbs_feature_names = self.model_config[nb_type_name+'_part_sparse_feature_names']
                features_embedding_list = self.get_features_embedding_list(nbs_feature_names, ad_nbs_features_embedding_dict[nb_type_name])
                ad_nbs_features_embedding_mapped_dict[nb_type_name] = self.layers_dict[this_scope_name][mapping_layer_name].inference(
                    tf.concat(features_embedding_list, 1)
                    )
                for nb_2_type_name in self.model_config["nb_config_dict_dict"][nb_type_name]:          
                    mapping_layer_name = nb_2_type_name+'_mapping'
                    nbs_2_feature_names = self.model_config[nb_2_type_name+'_part_sparse_feature_names']
                    features_embedding_list = self.get_features_embedding_list(nbs_2_feature_names, ad_nbs_2_features_embedding_dict[nb_type_name][nb_2_type_name])
                    features_embedding = tf.concat(features_embedding_list, 1)
                    ad_nbs_2_features_embedding_mapped_dict[nb_type_name][nb_2_type_name] = self.layers_dict[this_scope_name][mapping_layer_name].inference(
                        features_embedding
                        )
                
        # edge-level attention layer, aggregate neighbors within 2 hops
        if model_option[option_use_graph_data]:
            this_scope_name = "graph_feature_nbs_aggregation"
            with tf.variable_scope(this_scope_name, reuse=tf.AUTO_REUSE) as scope:
                self._build_graph_feature_nbs_aggregation(this_scope_name, option_use_semantic_attention)
                q_nbs_features_embedding_aggregated_dict = {}
                for nb_type_name in self.model_config["q_nb_type_names"]:
                    layer_name = "q_"+nb_type_name+"_nbs_2"
                    context_emb = self.add_context_feature(q_nbs_features_embedding_mapped_dict[nb_type_name], q_node_features_embedding_mapped, nbs_count=nbs_cnt)
                    q_nbs_2_features_embedding_mapped_list = self.dict_2_list(q_nbs_2_features_embedding_mapped_dict[nb_type_name])

                    q_nbs_features_embedding_aggregated_dict[nb_type_name] = self.layers_dict[this_scope_name][layer_name].inference(
                                            context_emb, 
                                            q_nbs_features_embedding_mapped_dict[nb_type_name],
                                            q_nbs_2_features_embedding_mapped_list
                                            )
                    q_nbs_features_embedding_aggregated_dict[nb_type_name] = tf.layers.batch_normalization(
                                                        q_nbs_features_embedding_aggregated_dict[nb_type_name], axis=-1, training=training)
                    
                    q_nbs_features_embedding_aggregated_dict[nb_type_name] = tfprint.tf_print_dense(q_nbs_features_embedding_aggregated_dict[nb_type_name], 
                                                                                                        "q_nbs_embedding_"+nb_type_name+"_aggregated", first_n=20, summarize=256)
                u_nbs_features_embedding_aggregated_dict = {}
                for nb_type_name in self.model_config["u_nb_type_names"]:
                    layer_name = "u_"+nb_type_name+"_nbs_2"
                    context_emb = self.add_context_feature(u_nbs_features_embedding_mapped_dict[nb_type_name], u_node_features_embedding_mapped, nbs_count=nbs_cnt)
                    u_nbs_2_features_embedding_mapped_list = self.dict_2_list(u_nbs_2_features_embedding_mapped_dict[nb_type_name])

                    u_nbs_features_embedding_aggregated_dict[nb_type_name] = self.layers_dict[this_scope_name][layer_name].inference(
                                            context_emb,
                                            u_nbs_features_embedding_mapped_dict[nb_type_name],
                                            u_nbs_2_features_embedding_mapped_list
                                            )
                    u_nbs_features_embedding_aggregated_dict[nb_type_name] = tf.layers.batch_normalization(
                                                        u_nbs_features_embedding_aggregated_dict[nb_type_name], axis=-1, training=training)
                    
                    u_nbs_features_embedding_aggregated_dict[nb_type_name] = tfprint.tf_print_dense(u_nbs_features_embedding_aggregated_dict[nb_type_name], 
                                                                                                        "u_nbs_embedding_"+nb_type_name+"_aggregated", first_n=20, summarize=256)
                ad_nbs_features_embedding_aggregated_dict = {}
                for nb_type_name in self.model_config["i_nb_type_names"]:
                    layer_name = "i_"+nb_type_name+"_nbs_2"
                    context_emb = self.add_context_feature(ad_nbs_features_embedding_mapped_dict[nb_type_name], ad_node_features_embedding_mapped, nbs_count=nbs_cnt)
                    ad_nbs_2_features_embedding_mapped_list = self.dict_2_list(ad_nbs_2_features_embedding_mapped_dict[nb_type_name])

                    ad_nbs_features_embedding_aggregated_dict[nb_type_name] = self.layers_dict[this_scope_name][layer_name].inference(
                                            context_emb,
                                            ad_nbs_features_embedding_mapped_dict[nb_type_name],
                                            ad_nbs_2_features_embedding_mapped_list
                                            )
                    ad_nbs_features_embedding_aggregated_dict[nb_type_name] = tf.layers.batch_normalization(
                                                        ad_nbs_features_embedding_aggregated_dict[nb_type_name], axis=-1, training=training)
                   
                    ad_nbs_features_embedding_aggregated_dict[nb_type_name] = tfprint.tf_print_dense(ad_nbs_features_embedding_aggregated_dict[nb_type_name], 
                                                                                                        "ad_nbs_embedding_"+nb_type_name+"_aggregated", first_n=20, summarize=256)

                if model_option[option_input_query]:
                    context_emb = tf.add(q_node_features_embedding_mapped, u_node_features_embedding_mapped)
                    q_nbs_features_embedding_aggregated_list = self.dict_2_list(q_nbs_features_embedding_aggregated_dict)

                    q_nbs_embedding = self.layers_dict[this_scope_name]["q_nbs"].inference(
                        context_emb,
                        q_node_features_embedding_mapped, 
                        q_nbs_features_embedding_aggregated_list
                        )
                if model_option[option_input_user]:
                    context_emb = tf.add(u_node_features_embedding_mapped, q_node_features_embedding_mapped)
                    u_nbs_features_embedding_aggregated_list = self.dict_2_list(u_nbs_features_embedding_aggregated_dict)

                    u_nbs_embedding = self.layers_dict[this_scope_name]["u_nbs"].inference(
                        context_emb,
                        u_node_features_embedding_mapped,
                        u_nbs_features_embedding_aggregated_list
                        )
                if model_option[option_input_ad]:
                    context_emb = tf.add(ad_node_features_embedding_mapped, u_node_features_embedding_mapped)
                    ad_nbs_features_embedding_aggregated_list = self.dict_2_list(ad_nbs_features_embedding_aggregated_dict)

                    ad_nbs_embedding = self.layers_dict[this_scope_name]["i_nbs"].inference(
                        ad_node_features_embedding_mapped, 
                        # context_emb,
                        ad_node_features_embedding_mapped, 
                        ad_nbs_features_embedding_aggregated_list
                        )

        q_nbs_embedding = tfprint.tf_print_dense(q_nbs_embedding, "q_nbs_embedding", first_n=20, summarize=256)
        u_nbs_embedding = tfprint.tf_print_dense(u_nbs_embedding, "u_nbs_embedding", first_n=20, summarize=256)
        ad_nbs_embedding = tfprint.tf_print_dense(ad_nbs_embedding, "ad_nbs_embedding", first_n=20, summarize=256)

        qu_nbs_embedding = tf.concat([q_nbs_embedding, u_nbs_embedding], 1)

        qu_nbs_embedding = tf_ops.get_normed_tensor(qu_nbs_embedding, axis=1)
        ad_nbs_embedding = tf_ops.get_normed_tensor(ad_nbs_embedding, axis=1)

        qu_nbs_embedding = tfprint.tf_print_dense(qu_nbs_embedding, "qu_nbs_embedding_normed", first_n=20, summarize=256)
        ad_nbs_embedding = tfprint.tf_print_dense(ad_nbs_embedding, "ad_nbs_embedding_normed", first_n=20, summarize=256)
        #########

        if model_option[option_use_request_aggregation]:
            this_scope_name = "request_aggregation"
            with tf.variable_scope(this_scope_name, reuse=tf.AUTO_REUSE) as scope:
                self._build_request_aggregation(this_scope_name,
                                                request_qu=model_option[option_use_request_qu],
                                                request_ad=model_option[option_use_request_ad])

                if model_option[option_use_request_qu]:
                    request_qu_embedding = self.layers_dict[this_scope_name]["qu_self"].inference(
                        qu_nbs_embedding
                    )
                    # request_qu_embedding = self.layers_dict[this_scope_name]["qu_self"].inference(
                    #     [q_node_embedding, q_nbs_embedding, u_node_embedding, u_nbs_embedding]
                    # )
                    request_qu_embedding = tfprint.tf_print_dense(request_qu_embedding, "request_qu_embedding", first_n=20, summarize=256)
                    #########
                    request_qu_embedding = tf_ops.get_normed_tensor(request_qu_embedding, axis=1)

                if model_option[option_use_request_ad] and model_option[option_input_ad]:
                    request_ad_embedding = self.layers_dict[this_scope_name]["ad_self"].inference(
                        ad_nbs_embedding
                    )
                    request_ad_embedding = tfprint.tf_print_dense(request_ad_embedding, "request_ad_embedding", first_n=20, summarize=256)
                    #########
                    request_ad_embedding = tf_ops.get_normed_tensor(request_ad_embedding, axis=1)
                
                
        if model_option[option_task_model_clip_trace_qu]:
            self.add_trace_variables("request_qu_embedding", request_qu_embedding)
            return request_qu_embedding
        if model_option[option_task_trace_ad]:
            self.add_trace_variables("request_ad_embedding", request_ad_embedding)
            return request_ad_embedding
        if model_option[option_task_model_clip]:
            self.signature_outputs["request_qu_embedding"] = request_qu_embedding
            return request_qu_embedding
        if model_option[option_task_model_clip_test_auc]:
            request_ad_embedding = prefetch_data["request_ad_embedding"]
            # self.add_trace_variables("request_ad_embedding", request_ad_embedding)

        if model_option[option_use_request_score]:
            this_scope_name = "request_score"
            with tf.variable_scope(this_scope_name, reuse=tf.AUTO_REUSE) as scope:

                request_qu_embedding = tfprint.tf_print_dense(request_qu_embedding, "request_qu_embedding_normed", first_n=20, summarize=256)
                request_ad_embedding = tfprint.tf_print_dense(request_ad_embedding, "request_ad_embedding_normed", first_n=20, summarize=256)
                #########

                cosine_ensemble_dict = {}
                cosine_click = self.get_request_score_or_list(request_qu_embedding, request_ad_embedding,
                                                              score_type="inner_product")
                cosine_ensemble_dict["cosine_click"] = cosine_click


        return prefetch_data["labels"], cosine_ensemble_dict

    def get_hit_rate(self, cosine_click_list, labels, k_num):
        i = 0
        labels = tf.reduce_mean(labels, axis=-1, keep_dims=False)
        labels = tfprint.tf_print_dense(labels, "labels", first_n=20, summarize=128)
        flag_list = []
        tmp_1 = tf.ones([k_num], dtype=tf.float32, name=None)
        tmp_0 = tf.zeros([k_num], dtype=tf.float32, name=None)
        for cosine_click in cosine_click_list:
            cosine_click = tf.reshape(cosine_click, [-1])
            idx = tf.nn.top_k(cosine_click, k_num)[1]
            tmp = tf.where(tf.equal(idx, i), tmp_1, tmp_0)
            tmp_sum = tf.reduce_sum(tmp)
            flag_list.append(tmp_sum)
            i = i + 1
        flag = tf.convert_to_tensor(flag_list)
        hit = flag * labels
        label_sum = tf.reduce_sum(labels)
        hit_sum = tf.reduce_sum(hit)
        label_sum = tfprint.tf_print_dense(label_sum, "label_sum", first_n=20, summarize=128)
        hit_sum = tfprint.tf_print_dense(hit_sum, "hit_sum", first_n=20, summarize=128)
        # hit_rate = hit_sum/label_sum
        return label_sum, hit_sum



    def loss(self, labels, cosine_ensemble_dict):
        # cosine_sim_type = self.cosine_sim_type
        cosine_sim_type = self.cosine_sim_type = "sigmoid"
        tfprint.simple_print("zoomer.loss()",
                             "loss.cosine_sim_type={}, self.cosine_sim_type={}".format(cosine_sim_type, self.cosine_sim_type))
        # sigmoid_weight = self.sigmoid_weight # 3.0, 4.0, 5.0
        sigmoid_weight = self.sigmoid_weight = 4.0
        tfprint.simple_print("zoomer.loss()",
                             "loss.sigmoid_weight={}, self.sigmoid_weight={}".format(sigmoid_weight, self.sigmoid_weight))

        # # for sigmoid_weight = 4.0

        focal_weight = 2.0  # 0.5, 1.0, 2.0

        reg_loss_weight = self.model_config["reg_loss_weight"] #1e-8
        #reg_loss_weight = 0.0

        cosine_click = cosine_ensemble_dict["cosine_click"]

        sim_loss = loss_ops.cosine_focal_cross_entropy_loss_with_labels(cosines=cosine_click, labels=labels,
                                                                        cosine_sim_type=cosine_sim_type,
                                                                        sigmoid_weight=sigmoid_weight,
                                                                        focal_weight=focal_weight)

        sim_loss = tf.reduce_mean(sim_loss, name="sim_loss")
        self.add_metrics("sim_loss", sim_loss)


        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.add_metrics("reg_loss", reg_loss)

        loss_all = sim_loss + \
                   reg_loss * reg_loss_weight
        loss_all = 10*loss_all
        self.add_metrics("loss_all", loss_all)
        return loss_all

    def loss_test(self, labels, cosine_ensemble_dict):
        # cosine_sim_type = self.cosine_sim_type
        cosine_sim_type = self.cosine_sim_type = "sigmoid"
        tfprint.simple_print("zoomer.loss_test()",
                             "loss_test.cosine_sim_type={}, self.cosine_sim_type={}".format(cosine_sim_type,
                                                                                           self.cosine_sim_type))
        # sigmoid_weight = self.sigmoid_weight # 3.0, 4.0, 5.0
        sigmoid_weight = self.sigmoid_weight = 4.0
        tfprint.simple_print("zoomer.loss_test()",
                             "loss_test.sigmoid_weight={}, self.sigmoid_weight={}".format(sigmoid_weight,
                                                                                         self.sigmoid_weight))

        focal_weight = 1.0  # 0.5, 1.0, 2.0

        reg_loss_weight = self.model_config["reg_loss_weight"] #1e-8

        tfprint.simple_print("zoomer.loss_test()",
                             "focal_weight={},reg_loss_weight={}".format(focal_weight, reg_loss_weight)
                             )

        cosines = cosine_ensemble_dict["cosine_click"]
        probs = tf_utils.to_sim_score(cosines, sim_type=cosine_sim_type, sigmoid_weight=sigmoid_weight)
        self.add_trace_variables("cosine", cosines)
        self.add_trace_variables("predict_prob", probs)

        sim_loss = loss_ops.cosine_focal_cross_entropy_loss_with_labels(cosines=cosines, labels=labels,
                                                                        cosine_sim_type=cosine_sim_type,
                                                                        sigmoid_weight=sigmoid_weight,
                                                                        focal_weight=focal_weight)
        sim_loss = tf.reduce_mean(sim_loss, name="sim_loss")
        self.add_metrics("sim_loss", sim_loss)

        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.add_metrics("reg_loss", reg_loss)

        loss_all = sim_loss + \
                   reg_loss * reg_loss_weight
        self.add_metrics("loss_all", loss_all)

        return loss_all

    def auc_train(self, labels, cosine_ensemble_dict):
        cosine_sim_type = self.cosine_sim_type 
        tfprint.simple_print("zoomer.auc_train()",
                             "auc_train.cosine_sim_type={}, self.cosine_sim_type={}".format(cosine_sim_type, self.cosine_sim_type))
        sigmoid_weight = self.sigmoid_weight
        tfprint.simple_print("zoomer.auc_train()",
                             "auc_train.sigmoid_weight={}, self.sigmoid_weight={}".format(sigmoid_weight, self.sigmoid_weight))

        cosine_click = cosine_ensemble_dict["cosine_click"]
        cosine_click = tfprint.tf_print_dense(cosine_click, "cosine_click", first_n=20, summarize=128)
        auc_value, auc_update_op = auc_ops.cosine_auc_with_labels(cosines=cosine_click, labels=labels,
                                                                        cosine_sim_type=cosine_sim_type, sigmoid_weight=sigmoid_weight,
                                                                        num_thresholds=self.context.get("auc_bucket_num"))
        auc_value = tfprint.tf_print_dense(auc_value, "auc_value", first_n=20, summarize=128)
        self.add_metrics("train_auc", auc_value)

        auc_mean = auc_value
        auc_update_ops = [auc_update_op]

        return auc_mean, auc_update_ops

    def auc_test(self, labels, cosine_ensemble_dict):
        cosine_sim_type = self.cosine_sim_type
        tfprint.simple_print("zoomer.auc_test()",
                             "auc_test.cosine_sim_type={}, self.cosine_sim_type={}".format(cosine_sim_type, self.cosine_sim_type))
        sigmoid_weight = self.sigmoid_weight
        tfprint.simple_print("zoomer.auc_test()",
                             "auc_test.sigmoid_weight={}, self.sigmoid_weight={}".format(sigmoid_weight, self.sigmoid_weight))
        

        cosines = cosine_ensemble_dict["cosine_click"]
        cosines = tfprint.tf_print_dense(cosines, "cosine_click", first_n=20, summarize=128)
        #cosines = tfprint.tf_print_dense(cosines,)

        auc_value, auc_update_op = auc_ops.cosine_auc_with_labels(cosines=cosines, labels=labels,
                                                                  cosine_sim_type=cosine_sim_type, sigmoid_weight=sigmoid_weight,
                                                                  num_thresholds=self.context.get("auc_bucket_num"))
        self.add_metrics("test_auc", auc_value)

        auc_mean = auc_value
        auc_update_ops = [auc_update_op]

        return auc_mean, auc_update_ops




class FeatureAggregationLayer(object):
    """
    input: features embedding list
    """
    def __init__(self, layer_name='FeatureAggregationLayer', features_embed_list=None, features_dim_list=None,
                 input_dim=None, mid_dim=None, output_dim=None, init_type=None, init_para=None):
        self.layer_name = layer_name
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.output_dim = output_dim
        self.init_type = init_type
        self.init_para = init_para
        self.features_embed_list = features_embed_list
        self.features_dim_list = features_dim_list

        if self.mid_dim == None:
            self.mid_dim = self.output_dim

        self.build_layer()

        # if self.features_embed_list != None:
        #     self.init_outputs = self.inference(self.features_embed_list)

    def build_layer(self):
        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            print("variable_scope: " + tf.get_variable_scope().name + "    ,  reuse: ", tf.get_variable_scope().reuse)
            self.layer1 = dnn_layers.DnnSingleLayer(layer_name='layer1',
                                                    input_dim=self.input_dim, output_dim=self.mid_dim,
                                                    weight_init_type=self.init_type, weight_init_para=self.init_para)
            self.layer2 = dnn_layers.DnnSingleLayer(layer_name='layer2',
                                                    input_dim=self.mid_dim, output_dim=self.output_dim,
                                                    weight_init_type=self.init_type, weight_init_para=self.init_para)

    def inference(self, features_embedding):
        """
        Returns: logits
        """
        tfprint.simple_print("FeatureAggregationLayer", "{}.inference()".format(self.layer_name))
        inputs = features_embedding
        outputs = self.layer1.inference(inputs)
        outputs = self.layer2.inference(outputs)
        return outputs

class FeatureNbsAggregationLayer(object):
    def __init__(self, layer_name='FeatureNbsAggregationLayer',
                 node_features_embed_list=None, nbs_features_embed_list_list=None,
                 node_features_dim_list=None, nbs_features_dim_list_list=None, nbs_count_list=None,
                 input_dim=None, mid_dim=None, output_dim=None, 
                 option_use_semantic_attention=None, init_type=None, init_para=None):
        self.layer_name = layer_name
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.output_dim = output_dim
        self.init_type = init_type
        self.init_para = init_para
        self.node_features_embed_list = node_features_embed_list
        self.node_features_dim_list = node_features_dim_list
        self.nbs_features_embed_list_list = nbs_features_embed_list_list
        self.nbs_features_dim_list_list = nbs_features_dim_list_list
        self.nbs_count_list = nbs_count_list
        self.option_use_semantic_attention = option_use_semantic_attention

        self.nbs_dim_sum_list = [sum(v) for v in self.nbs_features_dim_list_list]
        if self.input_dim == None:
            self.input_dim = sum(self.node_features_dim_list) + sum(self.nbs_dim_sum_list)
        if self.mid_dim == None:
            self.mid_dim = self.output_dim

        self.build_layer()

        # if self.node_features_embed_list != None and self.nbs_features_embed_list_list != None:
        #     self.init_outputs = self.inference(self.node_features_embed_list, self.nbs_features_embed_list_list)

    def get_coefs(self, context, emb):  #[B, E], [B, cnt, E]
        context = tf.expand_dims(context, 1)    #[B, 1, 128]
        logits = tf.matmul(context, emb, transpose_a=False, transpose_b=True)   #[B, 1, 5]
        logits = tf.nn.softmax(tf.nn.leaky_relu(logits))
        return logits

    def get_att_aggregated_emb(self, logits, emb):
        emb = tf.matmul(logits, emb)
        emb = tf.reduce_mean(emb, axis=1, keep_dims=False)
        return emb

    def get_semantic_agg_embedding(self, concat_list, context_emb):
        logits = []
        output_list = []
        emb_cnt = 0
        for emb in concat_list:
            emb_cnt = emb_cnt+1
            seq = tf.concat([emb, context_emb], axis=-1)
            logits.append(self.layer["semantic"].inference(seq))
        logits = tf.concat(logits, axis=-1)
        logits = tf.nn.softmax(tf.nn.leaky_relu(logits))
        feature_index = 0
        for emb in concat_list:
            emb = tf.transpose(emb)*logits[:, feature_index] 
            emb = tf.transpose(emb)*emb_cnt
            output_list.append(emb)
            feature_index = feature_index + 1
        outputs = tf.concat(output_list, axis=-1)
        return outputs



    def build_layer(self):
        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            print("variable_scope: " + tf.get_variable_scope().name + "    ,  reuse: ", tf.get_variable_scope().reuse)
            self.layer = {}
            self.layer["nb_agg"] = dnn_layers.DnnSingleLayer(layer_name='nb_agg',
                                                    input_dim=3*128, output_dim=self.output_dim,
                                                    weight_init_type=self.init_type, weight_init_para=self.init_para)
            self.layer["semantic"] = dnn_layers.DnnSingleLayer(layer_name='semantic',
                                                    input_dim=256, output_dim=1,
                                                    weight_init_type=self.init_type, weight_init_para=self.init_para)


    def inference(self, context_emb, node_features_embed, nbs_features_embed_list):
        """
        Returns: logits
        """
        tfprint.simple_print("FeatureNbsAggregationLayer", "{}.inference()".format(self.layer_name))
        concat_list = [node_features_embed]
        num_top = 5
        for nb_count, nbs_features_embed in zip(self.nbs_count_list, nbs_features_embed_list):
            nb_embed_dim = nbs_features_embed.shape[1].value
            assert nb_embed_dim == 128, \
            "nb_embed_dim ERROR!!! nb_embed_dim={} !".format(nb_embed_dim)
            nb_embeds = nbs_features_embed
            nb_embeds = tf.reshape(nb_embeds, [-1, nb_count, 128])
            _coefs = self.get_coefs(context_emb, nb_embeds)
            _coefs = tf.reduce_mean(_coefs, axis=1, keep_dims=False)
            _coefs_top, _coefs_idx = tf.nn.top_k(_coefs, num_top, sorted=False)
            _coefs_top_sm = tf.nn.softmax(_coefs_top)
            _coefs_shape = tf.shape(_coefs)
            row_idx = tf.tile(tf.range(_coefs_shape[0])[:,tf.newaxis], (1, num_top))
            scatter_idx = tf.stack([row_idx, _coefs_idx], axis=-1)
            coefs = tf.scatter_nd(scatter_idx, _coefs_top_sm, _coefs_shape)
            coefs = tf.expand_dims(coefs, axis=1)
            coefs = tfprint.tf_print_dense(coefs, "coefs", first_n=10, summarize=32)
            nb_embeds = self.get_att_aggregated_emb(coefs, nb_embeds)
            concat_list.append(nb_embeds)
        if self.option_use_semantic_attention:
            inputs = self.get_semantic_agg_embedding(concat_list, context_emb)
        else:
            inputs = tf.concat(concat_list, axis=-1)
        outputs = self.layer["nb_agg"].inference(inputs)
        return outputs


class FeatureLevelAttention(object):

    def __init__(self, feature_dim_dict, layer_name='FeatureLevelAttentionLayer',
                init_type=None, init_para=None):
        self.layer_name = layer_name
        self.feature_dim_dict = feature_dim_dict
        self.init_type = init_type
        self.init_para = init_para

        self.build_layer()
    
    def build_layer(self):
        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            print("variable_scope: " + tf.get_variable_scope().name + "    ,  reuse: ", tf.get_variable_scope().reuse)
            self.layers = {}
            for feature_name in self.feature_dim_dict:
                self.layers[feature_name] = dnn_layers.DnnSingleLayer(layer_name=feature_name,
                                                    input_dim=self.feature_dim_dict[feature_name], output_dim=1,
                                                    weight_init_type=self.init_type, weight_init_para=self.init_para)

    def inference(self, input_dict, context_feature_dict, dict_depth, nb_cnt):
        """
        Returns: logits
        """
        tfprint.simple_print("FeatureLevelAttention", "{}.inference()".format(self.layer_name))
        output_dict = {}
        if dict_depth != 0:
            for name, sub_dict in input_dict.items():
                output_dict[name] = self.inference(sub_dict, context_feature_dict, dict_depth-1, nb_cnt)
            return output_dict
        else:
            logits = []
            context_feature_list = []
            feature_cnt=0
            for feature_name in context_feature_dict:
                context_feature_list.append(context_feature_dict[feature_name])
            context_feature = tf.concat(context_feature_list, axis=-1)
            context_feature_dim = context_feature.shape[-1].value
           
            context_feature = tf.expand_dims(context_feature, axis=1)
            context_feature = tf.tile(context_feature, [1, nb_cnt, 1])  #[cnt, nb_cnt, x]
            context_feature = tf.reshape(context_feature, [-1, context_feature_dim])    #[cnt*nb_cnt, x]
            for feature_name in input_dict:
                if feature_name == "query_id":
                    layer_index = "query_id"
                    assert input_dict[feature_name].shape[-1].value==64, \
                        "FeatureLevelAttention ERROR!!!dim={} ".format(input_dict[feature_name].shape[-1].value)
                elif feature_name == "item_id" or feature_name == "user_id":
                    layer_index = "iu_id"
                    assert input_dict[feature_name].shape[-1].value==32, \
                        "FeatureLevelAttention ERROR!!!dim={} ".format(input_dict[feature_name].shape[-1].value)
                else:
                    layer_index = "normal"
                    assert input_dict[feature_name].shape[-1].value==16, \
                        "FeatureLevelAttention ERROR!!!dim={} ".format(input_dict[feature_name].shape[-1].value)
                feature_cnt = feature_cnt + 1
                seq = tf.concat([input_dict[feature_name], context_feature], axis=-1)   #[cnt*nb_cnt, 16+x]
                logits.append(self.layers[layer_index].inference(seq))   #[cnt*nb_cnt, 1]
            logits = tf.concat(logits, axis=-1)  
            logits = tf.nn.softmax(tf.nn.leaky_relu(logits))    #[cnt*nb_cnt, feature_cnt]
            feature_index = 0
            for feature_name in input_dict:
                output_dict[feature_name] = tf.transpose(input_dict[feature_name])*logits[:, feature_index] 
                output_dict[feature_name] = tf.transpose(output_dict[feature_name])*feature_cnt
                feature_index = feature_index + 1
        return output_dict
