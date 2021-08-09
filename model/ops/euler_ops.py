# coding=utf-8

import tensorflow as tf
import tf_euler
import argparse
from xdl.utils import tfprint
from model import zoomer_model_config as zoomer_model_config

# def init_graph():
#     zk_addr = ctx.get_config("extend_role", 'euler', 'zk_addr')
#     zk_path = '/euler/{}'.format(ctx.get_app_id())
#     shard_num = ctx.get_config('extend_role', 'euler', 'shard_num')
#     tf_euler.initialize_graph({
#           'mode': 'remote',
#           'zk_server': zk_addr,
#           'zk_path': zk_path,
#           'shard_num': shard_num,
#           'num_retries':100
#       })
def init_graph(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument('--app_id', default=None)
    args, _ = parser.parse_known_args() # TODO: Use context.
    zk_addr = ctx.get("extend_role", 'euler', 'zk_addr')
    shard_num = ctx.get('extend_role', 'euler', 'shard_num')
    zk_path = '/euler/{}'.format(args.app_id)
    tf_euler.initialize_graph({
      'mode': 'remote',
      'zk_server': zk_addr,
      'zk_path': zk_path,
      'shard_num': shard_num,
      'num_retries': 100
    })


# http://gitlab.alibaba-inc.com/alimama-data-infrastructure/Euler-2.0/wikis/tf_op



def sample_by_walk(src, edge_type, walk_len=1, walk_p=1, walk_q=1):
    walk_path = [edge_type]
    walk_path.extend([['1'] for i in range(walk_len - 1)])
    path = tf_euler.random_walk(
        src, walk_path,
        p=walk_p,
        q=walk_q)

    path_s = tf.slice(path, [0,1], [-1, walk_len])

    pos = tf.reshape(path_s, [-1])

    #negs_s = tf.reshape(negs, [-1, 24])
    #negs_copy = tf.slice(negs_s, [0,0], [-1, walk_len * 6])

    negs_copy = tf_euler.sample_node_with_src(pos, 6 * walk_len)

    #negs_copy = tf.concat([negs for i in range(walk_len)], 1)
    src_s=tf.reshape(src, [-1, 1])
    src_copy = tf.concat([src_s for i in range(walk_len)], 1)

    #have some problems
    neg_nodes = tf.reshape(negs_copy, [-1])
    src_nodes = tf.reshape(src_copy, [-1])
    #neg_nodes = tf.reshape(negs_copy, [-1, 2])

    return src_nodes, pos, neg_nodes


def get_node_filled_dense_features(nodes, n_feature_names, dimensions):
    # nodes = tf.reshape(nodes, [-1])
    node_filled = tf_euler.get_dense_feature(nodes, n_feature_names, dimensions)
    return node_filled

def get_node_filled_sparse_features(nodes, n_feature_names):
    # nodes = tf.reshape(nodes, [-1])
    node_filled = tf_euler.get_sparse_feature(nodes, n_feature_names)
    return node_filled

def sample_node_neighbors(nodes, edge_types, nb_cnt):
    # nodes = tf.reshape(nodes, [-1])
    node_neighbors, _, _ = tf_euler.sample_neighbor(nodes, edge_types=edge_types, count=nb_cnt)
    # node_neighbors = tf.reshape(node_neighbors, [-1])
    return node_neighbors





def neg_sampling_with_type(targets, neg_num):
    # targets = tf.reshape(targets, [-1])
    negs = tf_euler.sample_node_with_src(targets, neg_num)
    # negs = tf.reshape(negs, [-1])
    return negs


def fetch_node_neighbors_and_sparse_features(nodes, n_feature_names, nb_config_dict):
    """
    n_feature_names: list
    nb_config_dict: {"nb_type_name": [nb_cnt, nb_edge_types, nb_feature_names]}
    """
    node_filled_features = get_node_filled_sparse_features(nodes, n_feature_names)
    node_filled_dict = dict(zip(n_feature_names, node_filled_features))
    nb_config_dict_dict = zoomer_model_config.nb_config_dict_dict
    nbs_filled_dict = {}
    nbs_2_filled_dict = {}
    for nb_type_name in nb_config_dict:
        nb_cnt, nb_edge_types, nb_feature_names = nb_config_dict[nb_type_name]
        node_neighbors = sample_node_neighbors(nodes, edge_types=nb_edge_types, nb_cnt=nb_cnt)
        node_neighbors = tf.reshape(node_neighbors, [-1])
        nbs_2_filled_dict[nb_type_name] = {}
        # node_neighbors = tfprint.tf_print_dense(node_neighbors, "node_neighbors", first_n=10, summarize=32)
        for nb_2_type_name in nb_config_dict_dict[nb_type_name]:
            nb_2_cnt, nb_2_edge_types, nb_2_feature_names = nb_config_dict_dict[nb_type_name][nb_2_type_name]
            node_neighbors_2 = sample_node_neighbors(node_neighbors, edge_types=nb_2_edge_types, nb_cnt=nb_2_cnt)
            node_neighbors_2 = tf.reshape(node_neighbors_2, [-1])
            nb_2_filled_features = get_node_filled_sparse_features(node_neighbors_2, nb_2_feature_names)
            nbs_2_filled_dict[nb_type_name][nb_2_type_name] = dict(zip(nb_2_feature_names, nb_2_filled_features))
        nb_filled_features = get_node_filled_sparse_features(node_neighbors, nb_feature_names)
        nbs_filled_dict[nb_type_name] = dict(zip(nb_feature_names, nb_filled_features))
    return node_filled_dict, nbs_filled_dict, nbs_2_filled_dict

def assemble_graph_sparse(nodes, n_sparse_feature_names=None, nb_config_dict=None):
    nodes = tf.reshape(nodes, [-1])
    node_filled_dict, nbs_filled_dict, nbs_2_filled_dict = fetch_node_neighbors_and_sparse_features(nodes,
                                                                                 n_sparse_feature_names,
                                                                                 nb_config_dict)
    return node_filled_dict, nbs_filled_dict, nbs_2_filled_dict


# def fetch_node_neighbors_and_sparse_features(nodes, n_feature_names, nb_config_dict):
#     """
#     n_feature_names: list
#     nb_config_dict: {"nb_type_name": [nb_cnt, nb_edge_types, nb_feature_names]}
#     """
#     node_filled_features = get_node_filled_sparse_features(nodes, n_feature_names)
#     node_filled_dict = dict(zip(n_feature_names, node_filled_features))
#     nbs_filled_dict = {}
#     for nb_type_name in nb_config_dict:
#         nb_cnt, nb_edge_types, nb_feature_names = nb_config_dict[nb_type_name]
#         node_neighbors = sample_node_neighbors(nodes, edge_types=nb_edge_types, nb_cnt=nb_cnt)
#         node_neighbors = tf.reshape(node_neighbors, [-1])
#         # node_neighbors = tfprint.tf_print_dense(node_neighbors, "node_neighbors", first_n=10, summarize=32)
#         nb_filled_features = get_node_filled_sparse_features(node_neighbors, nb_feature_names)
#         nbs_filled_dict[nb_type_name] = dict(zip(nb_feature_names, nb_filled_features))
#     return node_filled_dict, nbs_filled_dict

# def assemble_graph_sparse(nodes, n_sparse_feature_names=None, nb_config_dict=None):
#     nodes = tf.reshape(nodes, [-1])
#     node_filled_dict, nbs_filled_dict = fetch_node_neighbors_and_sparse_features(nodes,
#                                                                                  n_sparse_feature_names,
#                                                                                  nb_config_dict)
#     return node_filled_dict, nbs_filled_dict











def _sample_node_neighbors_and_features(node, n_feature_names, edge_types_list, nb_cnt, nb_feature_names_list):
    """
    input:
        node: tf.tensor, tf.int64, [batch_size].
        n_feature_names: list, string.
        edge_types_list: list[list], string. 
    output:
        node_filled: list of tf.sparseTensor, tf,int64, [[batch_size]], ]
    """
    node_filled = get_node_filled_sparse_features(node, n_feature_names)
    nb_filled_list = []
    for idx in range(len(edge_types_list)):
        nbs = sample_node_neighbors(node, edge_types = edge_types_list[idx], nb_cnt = nb_cnt)
        nbs_filled = get_node_filled_sparse_features(nbs, nb_feature_names_list[idx])
        nb_filled_list.append(nbs_filled)
    return node_filled, nb_filled_list

def _assemble_graph_n_neg_pretrain(sources, targets,
                                  q_full_features=None, i_full_features=None,
                                  neg_num=None, nb_cnt=None,
                                  q_nb_etypes_list=None, i_nb_etypes_list=None,
                                  q_nb_features_list=None, i_nb_features_list=None,
                                  ):
    # negative sampling
    sources = tf.reshape(sources, [-1])
    targets = tf.reshape(targets, [-1])
    negs = neg_sampling_with_type(targets, neg_num)
    src_q, src_q_nb_list = _sample_node_neighbors_and_features(sources, q_full_features, q_nb_etypes_list, nb_cnt,
                                                              q_nb_features_list)
    src_i, src_i_nb_list = _sample_node_neighbors_and_features(sources, i_full_features, i_nb_etypes_list, nb_cnt,
                                                              i_nb_features_list)
    tar_i, tar_i_nb_list = _sample_node_neighbors_and_features(targets, i_full_features, i_nb_etypes_list, nb_cnt,
                                                              i_nb_features_list)
    neg_i, neg_i_nb_list = _sample_node_neighbors_and_features(negs, i_full_features, i_nb_etypes_list, nb_cnt,
                                                              i_nb_features_list)

    return src_q, src_q_nb_list, src_i, src_i_nb_list, tar_i, tar_i_nb_list, neg_i, neg_i_nb_list

def _assemble_graph_with_negs(q_nodes, i_nodes, ad_nodes, neg_num=None,
                   q_full_features=None, i_full_features=None, nb_cnt=None,
                   q_nb_etypes_list=None, i_nb_etypes_list=None,
                   q_nb_features_list=None, i_nb_features_list=None,
                   ):
    q_nodes = tf.reshape(q_nodes, [-1])
    i_nodes = tf.reshape(i_nodes, [-1])
    ad_nodes = tf.reshape(ad_nodes, [-1])
    neg_nodes = neg_sampling_with_type(ad_nodes, neg_num)
    q_filled_nodes, q_filled_nodes_nb_list = _sample_node_neighbors_and_features(q_nodes, q_full_features, q_nb_etypes_list, nb_cnt,
                                                              q_nb_features_list)
    i_filled_nodes, i_filled_nodes_nb_list = _sample_node_neighbors_and_features(i_nodes, i_full_features, i_nb_etypes_list, nb_cnt,
                                                              i_nb_features_list)
    ad_filled_nodes, ad_filled_nodes_nb_list = _sample_node_neighbors_and_features(ad_nodes, i_full_features, i_nb_etypes_list, nb_cnt,
                                                                  i_nb_features_list)
    neg_filled_nodes, neg_filled_nodes_nb_list = _sample_node_neighbors_and_features(neg_nodes, i_full_features, i_nb_etypes_list, nb_cnt,
                                                                                  i_nb_features_list)


    return q_filled_nodes, q_filled_nodes_nb_list, i_filled_nodes, i_filled_nodes_nb_list, \
           ad_filled_nodes, ad_filled_nodes_nb_list, neg_filled_nodes, neg_filled_nodes_nb_list

def _assemble_graph(nodes, nb_cnt=None, node_full_features=None, node_nb_etypes_list=None, node_nb_features_list=None):
    nodes = tf.reshape(nodes, [-1])
    node_filled_nodes, node_filled_nodes_nb_list = _sample_node_neighbors_and_features(nodes, node_full_features,
                                                                                      node_nb_etypes_list, nb_cnt,
                                                                                      node_nb_features_list)
    return node_filled_nodes, node_filled_nodes_nb_list