# coding=utf-8
import tensorflow as tf


###################################################################################################
# common
###################################################################################################

embed_init_type = 1
embed_init_para = {
        "stddev": 0.02 # 0.01, 0.02, 0.05, 0.1
}
dense_init_type = 2
dense_init_para = {
        "factor": 0.4 # 0.2, 0.4, 0.8, 1.0, 1.2, 1.4  # linear: ~1.0, relu: ~1.43, tanh: ~1.15
}

model_dtype = tf.float32  # not used
embed_dtype = tf.float32  # not used
embed_hts_heuristic_strategy = False  # not used, context.meta_partitioner
embed_base_dim = 16
reg_loss_weight = 1e-6  # 1e-6, 1e-7, 1e-8,
focal_weight = 2.0  # 0.5, 1.0, 2.0  # not used
adagrad_learning_rate = 0.1 # 0.02  # not used


###################################################################################################
# model
###################################################################################################

nb_cnt = 5
nb_total_cnt = 10
nb_2_cnt = 5
neg_num = 10
neg_pv_num = 5
neg_unpv_num = 5
urb_click_seq_max_length = 50
fea_embed_dim = embed_base_dim
node_embed_dim = 64 # 64, 128
nbs_agg_embed_dim = 64 # 64, 128
node_agg_embed_dim = 64 # 64, 128
urb_embed_dim = 128
req_embed_dim = 128

mapped_out_dim = 128
space_mapping_mid_dim = 128
space_mapping_out_dim = 64

trans_out_dim = 64




###################################################################################################
# graph
###################################################################################################

node_all_sparse_features_dict = {
        # feature_name: {share_name/table: XXX, prefix: XXX, amount: XXX, varlen: max_dim, fixlen: min_dim}
        "query_id": {"table_name": "query_id",  "prefix": "query_id",  "amount":1055111996, "min_dim": 1, "max_dim": 1},
        "q_terms":  {"table_name": "term_word", "prefix": "term_word", "amount":5045058,    "min_dim": 1, "max_dim": 20},

        "item_id":       {"table_name": "item_id",   "prefix": "item_id",   "amount":55560248, "min_dim": 1, "max_dim": 1},
        "i_title_terms": {"table_name": "term_word", "prefix": "term_word", "amount":5045058,  "min_dim": 1, "max_dim": 20},
        "i_cate_leaf":   {"table_name": "cate_id",   "prefix": "cate_id",   "amount":30038,    "min_dim": 1, "max_dim": 1},
        "i_cate_root":   {"table_name": "cate_id",   "prefix": "cate_id",   "amount":300,      "min_dim": 1, "max_dim": 1},
        "i_brand_id":    {"table_name": "brand_id",  "prefix": "brand_id",  "amount":4045218,  "min_dim": 1, "max_dim": 1},
        "i_commod_id":   {"table_name": "commod_id", "prefix": "commod_id", "amount":0000,     "min_dim": 1, "max_dim": 1},
        "i_shop_id": {"table_name": "shop_id", "prefix": "shop_id", "amount":10381624, "min_dim": 1, "max_dim": 1},
        "i_city_nm": {"table_name": "city_nm", "prefix": "city_nm", "amount":0000, "min_dim": 1, "max_dim": 1},
        "i_prov_nm": {"table_name": "prov_nm", "prefix": "prov_nm", "amount":0000, "min_dim": 1, "max_dim": 1},
        "i_props":   {"table_name": "i_prop",  "prefix": "i_prop",  "amount":0000, "min_dim": 1, "max_dim": 20},
        "i_tags":    {"table_name": "i_tag",   "prefix": "i_tag",   "amount":0000, "min_dim": 1, "max_dim": 20},

        "user_id":  {"table_name": "user_id", "prefix": "user_id", "amount":233555855, "min_dim": 1, "max_dim": 1},
        "u_age":    {"table_name": "age",     "prefix": "age",     "amount":0000,      "min_dim": 1, "max_dim": 1},
        "u_gender": {"table_name": "gender",  "prefix": "gender",  "amount":0000,      "min_dim": 1, "max_dim": 1},
        "u_occupation": {"table_name": "occupation",  "prefix": "occupation", "amount":0000, "min_dim": 1, "max_dim": 1},
        "u_prov_id":       {"table_name": "prov_id",  "prefix": "prov_id",    "amount":0000, "min_dim": 1, "max_dim": 1},
        "u_city_id":       {"table_name": "city_id",  "prefix": "city_id",    "amount":0000, "min_dim": 1, "max_dim": 1},
        "u_star_level":    {"table_name": "u_starl",  "prefix": "u_starl",    "amount":0000, "min_dim": 1, "max_dim": 1},
        "u_vip_level":     {"table_name": "u_vipl",   "prefix": "u_vipl",     "amount":0000, "min_dim": 1, "max_dim": 1},
        "u_perfer_cates":  {"table_name": "cate_id",  "prefix": "cate_id",    "amount":0000, "min_dim": 1, "max_dim": 20},
        "u_perfer_brands": {"table_name": "brand_id", "prefix": "brand_id",   "amount":0000, "min_dim": 1, "max_dim": 20},

}
q_full_sparse_feature_names = ["query_id", "q_terms"]
i_full_sparse_feature_names = ["item_id", "i_title_terms", "i_cate_leaf", "i_cate_root", "i_brand_id", "i_commod_id", "i_shop_id",
                               "i_city_nm", "i_prov_nm",
                               ]
u_full_sparse_feature_names = ["user_id", "u_age", "u_gender", "u_occupation", "u_prov_id", "u_city_id", "u_star_level", "u_vip_level",
                               "u_perfer_cates", "u_perfer_brands",
                               ]


qiu_full_sparse_feature_names  = ["query_id", "q_terms", 
                                  "item_id", "i_title_terms", "i_cate_leaf", "i_cate_root", "i_brand_id", "i_commod_id", "i_shop_id",
                               "i_city_nm", "i_prov_nm", 
                               "user_id", "u_age", "u_gender", "u_occupation", "u_prov_id", "u_city_id", "u_star_level", "u_vip_level",
                               "u_perfer_cates", "u_perfer_brands",
                               ]


q_part_sparse_feature_names = q_full_sparse_feature_names
i_part_sparse_feature_names = i_full_sparse_feature_names
u_part_sparse_feature_names = u_full_sparse_feature_names


node_type_list = [
        "query_id", "item_id", "user_id",
        "__EMPTY__", "__DEFAULT__"
]
edge_type_list = [
        "q-i", "q-i-q",
        "i-q", "i-q-i",
        "u-q", "u-i",
]

# nb_config_dict: {"nb_type_name": [nb_cnt, nb_edge_types, nb_feature_names]}

q_nb_config_dict = {
        "q": [nb_cnt, ["q-i-q"], q_part_sparse_feature_names],
        "i": [nb_cnt, ["q-i"], i_part_sparse_feature_names],
}
q_nb_type_names = ["q", "i"]
i_nb_config_dict = {
        "q": [nb_cnt, ["i-q"], q_part_sparse_feature_names],
        "i": [nb_cnt, ["i-q-i"], i_part_sparse_feature_names],
}
i_nb_type_names = ["q", "i"]
u_nb_config_dict = {
        "q": [nb_cnt, ["u-q"], q_part_sparse_feature_names],
        "i": [nb_cnt, ["u-i"], i_part_sparse_feature_names],
}
u_nb_type_names = ["q", "i"]


nb_config_dict_dict = {
        "q": q_nb_config_dict,
        "i": i_nb_config_dict,
        "u": u_nb_config_dict,
}

feature_dim_dict = {
    "normal": 192,
    "query_id": 240,
    "iu_id": 208,
}
'''
q_nbs_config_dict = {
        "edge_types": [["q-i", "q-i-q", "", "", "", ""], edge_type_list],
        "counts": [nb_cnt, nb_2_cnt],
}

i_nbs_config_dict = {
        "edge_types": [["i-q", "i-q-i", "", "", "", ""], edge_type_list],
        "counts": [nb_cnt, nb_2_cnt],
}

u_nbs_config_dict = {
        "edge_types": [["u-i", "u-i", "", "", "", ""], edge_type_list],
        "counts": [nb_cnt, nb_2_cnt],
}
'''

###################################################################################################
# input context
###################################################################################################

urb_sparse_feature_names = ["preitem", "preshop", "prebrand", "precate", "prerootcate"]





###################################################################################################
# embedding config table
###################################################################################################

all_sparse_embed_hts_configs_dict = {
        # embedding_hash_table_name: {dim, dtype, init_type, init_para, heuristic_strategy, trainable, combiner, norm, align}

        "query_id":  {"dim": embed_base_dim*4, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "term_word": {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},

        "item_id":       {"dim": embed_base_dim*2, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "cate_id":       {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "brand_id":  {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "commod_id": {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "shop_id":   {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "i_prop":    {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "i_tag":     {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "prov_nm":   {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "city_nm":   {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},

        "user_id":   {"dim": embed_base_dim*2, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "age":       {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "gender":    {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "occupation":{"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "u_starl":   {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "u_vipl":    {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "prov_id":   {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        "city_id":   {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        # "u_perfer_cates":  {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},
        # "u_perfer_brands": {"dim": embed_base_dim, "init_type": embed_init_type, "init_para": embed_init_para, "combiner": "mean"},

}

all_aggregation_embed_hts_configs_dict = {
        "q_node_embedding": {"dim": node_embed_dim, "init_type": embed_init_type, "init_para": embed_init_para,
                             "combiner": "mean", "heuristic_strategy": False}, # 暂时只支持fixed
        "q_nbs_embedding": {"dim": nbs_agg_embed_dim, "init_type": embed_init_type, "init_para": embed_init_para,
                            "combiner": "mean", "heuristic_strategy": False}, # 暂时只支持fixed
        "u_node_embedding": {"dim": node_embed_dim, "init_type": embed_init_type, "init_para": embed_init_para,
                             "combiner": "mean", "heuristic_strategy": False},  # 暂时只支持fixed
        "u_nbs_embedding": {"dim": nbs_agg_embed_dim, "init_type": embed_init_type, "init_para": embed_init_para,
                            "combiner": "mean", "heuristic_strategy": False},  # 暂时只支持fixed
}
# query_aggregation_embed_table_names = ["q_node_embedding", "q_nbs_embedding"]
# user_aggregation_embed_table_names = ["u_node_embedding", "u_nbs_embedding"]




###################################################################################################
###################################################################################################
option_task_trace_tmp = "task_trace_tmp"
option_task_test_auc = "task_test_auc"
option_task_trace_ad = "task_trace_ad"
option_task_traceM_query = "task_traceM_query"
option_task_traceM_user = "task_traceM_user"
option_task_model_clip = "task_model_clip"
option_task_model_clip_test_auc = "task_model_clip_test_auc"
option_task_model_clip_trace_qu = "task_model_clip_trace_qu"
option_input_sample_id = "input_sample_id"
option_input_unseen_flag = "input_unseen_flag"
option_input_labels = "input_labels"
option_input_query = "input_query"
option_input_user = "input_user"
option_input_ad = "input_ad"
option_input_ad_neg = "input_ad_neg"
option_input_context = "input_context"
option_input_request_ad_embedding = "input_request_ad_embedding"
option_use_graph_data = "use_graph_data"
option_use_data_prefetch = "use_data_prefetch"
option_use_aggregation_embedding = "use_aggregation_embedding"
option_use_request_aggregation = "use_request_aggregation"
option_use_request_qu = "use_request_qu"
option_use_request_ad = "use_request_ad"
option_use_request_score = "use_request_score"
option_use_feature_level_attention = "use_feature_level_attention"
option_use_semantic_attention = "use_semantic_attention"