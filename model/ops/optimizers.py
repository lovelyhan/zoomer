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

def refresh_props(props):
    default_props = \
        {
            "learning_rate": 0.01,
            "lr_decay_rate": 1,

            "Momentum": {
                "momentum": 0.9,
                "use_nesterov": False
            },
            "Adagrad": {
                "initial_accumulator_value": 0.1
            },
            "AdagradDecay": {
                "initial_accumulator_value": 0.1,
                "initial_global_step": 0,
                "decay_rate": 0.95,
                "interval_step": 50000
            },
            "Adadelta": {
                "rho": 0.9,
                "epsilon": 1e-10
            },
            # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/optimizers/RMSprop?hl=zh-cn
            "RMSProp": {
                "decay": 0.9,
                "momentum": 0, # 0.5, 0.1, 0.9
                "epsilon": 1e-10
            },
            "Adam": {
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-08
                # "lr_decay": True
            },
            "Ftrl": {
                "learning_rate_power": -0.5,
                "initial_accumulator_value": 0.1,
                "l1_regularization": 0.0,
                "l2_regularization": 0.0
            }
        }
    _props = {}
    _props.update(default_props)
    _props.update(props)
    return _props

def get_optval(opt_name, key, props):
    val = None
    if key in props:
      val = props[key]
    elif key in props[opt_name]:
      val = props[opt_name][key]
    return val


def create_optimizer_by_props(props):
    learning_rate = props["learning_rate"]
    lr_decay_rate = props["lr_decay_rate"]
    if lr_decay_rate != 1:
        raise ValueError("lr_decay_rate should be 1")

    optimizer = None

    opt_name = props["name"]

    if opt_name == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                 momentum=get_optval(opt_name, 'momentum', props),
                                 use_nesterov=get_optval(opt_name, 'use_nesterov', props))
    elif opt_name == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif opt_name == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                initial_accumulator_value=get_optval(opt_name, 'initial_accumulator_value', props))
    # elif opt_name == 'AdagradDecay':
    #     optimizer = tf.train.AdagradDecay(learning_rate=learning_rate,
    #                                  initial_accumulator_value=get_optval(opt_name, 'initial_accumulator_value', props),
    #                                  initial_global_step=get_optval(opt_name, 'initial_global_step', props),
    #                                  decay_rate=get_optval(opt_name, 'decay_rate', props),
    #                                  interval_step=get_optval(opt_name, 'interval_step', props))
    elif opt_name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                             beta1=get_optval(opt_name, 'beta1', props),
                             beta2=get_optval(opt_name, 'beta2', props),
                             epsilon=float(get_optval(opt_name, 'epsilon', props)))
    elif opt_name == 'Ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate=get_optval(opt_name, 'learning_rate', props),
                             learning_rate_power=get_optval(opt_name, 'learning_rate_power', props),
                             initial_accumulator_value=get_optval(opt_name, 'initial_accumulator_value', props),
                             l1_regularization_strength=get_optval(opt_name, 'l1_regularization', props),
                             l2_regularization_strength=get_optval(opt_name, 'l2_regularization', props))
    elif opt_name == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                decay=get_optval(opt_name, 'decay', props),
                                momentum=get_optval(opt_name, 'momentum', props),
                                epsilon=get_optval(opt_name, 'epsilon', props))
    elif opt_name == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                     rho=get_optval(opt_name, 'rho', props),
                                     epsilon=get_optval(opt_name, 'epsilon', props))

    if optimizer is None:
        raise ValueError('optimizer not support, ' + str(props))

    return optimizer

def get_optimizers(ctx, config):
    tfprint.simple_print_all("optimizers config", config, highlight=True)
    SyncReplicasOpt = config["SyncReplicasOpt"]
    props = config["global_optimizer"]
    if props is None:
        raise ValueError("global_optimizer is None")
    props = refresh_props(props)
    global_optimizer = create_optimizer_by_props(props)

    # TODO : scope_optimizer
    # optimizer_scopes = {}
    # if 'scope_optimizer' in config:
    #     for props in config['scope_optimizer']:
    #         props = refresh_props(props)
    #
    #         scope_list = []
    #         scopes = props['scopes']
    #         for scope in scopes.split(','):
    #             scope = scope.strip()
    #             if scope == '':
    #                 continue
    #             scope_list.append(scope)
    #             print('scope [%s] optimizer config [%s]' % (scope, str(props)))
    #         if len(scope_list) == 0:
    #             continue
    #         optimizer_scopes[create_optimizer_by_props(props)] = scope_list

    sync_replicas_hook = None
    if SyncReplicasOpt:
        tfprint.simple_print_all("SyncReplicasOpt", SyncReplicasOpt, highlight=True)
        global_optimizer = tf.train.SyncReplicasOptimizer(
            global_optimizer,
            replicas_to_aggregate=ctx.worker_num,
            total_num_replicas=ctx.worker_num,
            use_locking=True,
            sparse_accumulator_type='hash',
            sparse_reduction_type='CMEAN')
        sync_replicas_hook = global_optimizer.make_session_run_hook(ctx.is_chief(), 0)

    return global_optimizer, sync_replicas_hook

