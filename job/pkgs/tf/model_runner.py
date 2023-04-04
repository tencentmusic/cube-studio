# coding=utf-8
# @Time     : 2020/9/22 13:19
# @Auther   : lionpeng@tencent.com

import copy
import json
import os
import subprocess
import sys
import time
import traceback
import types
from typing import Union

import tensorflow as tf
from tensorflow.python.eager import backprop, context
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import data_adapter

from .helperfuncs import TF_REF_VERSION, tf_log_level_from_string

if tf.__version__ < TF_REF_VERSION:
    try:
        from tensorflow.python.keras.engine.training import (
            _keras_api_gauge, enable_multi_worker)
    except:
        from tensorflow.python.keras.engine.base_layer import \
            keras_api_gauge as _keras_api_gauge
        def enable_multi_worker(worker_fn):
            return worker_fn

    try:
        from tensorflow.python.keras.engine.training import \
            _disallow_inside_tf_function
    except:
        print("tf _disallow_inside_tf_function not exists, ignore it", flush=True)
        def _disallow_inside_tf_function(*args):
            pass
else:
    from tensorflow.python.keras.engine.base_layer import \
        keras_api_gauge as _keras_api_gauge
    from tensorflow.python.keras.engine.training import \
        _disallow_inside_tf_function

    def enable_multi_worker(worker_fn):
        return worker_fn
from tensorflow.python.keras.utils import tf_utils, version_utils

try:
    from tensorflow.python.profiler import trace
    profile_trace = trace.Trace
except:
    print("using traceme.TraceMe as tf profile event tracer", flush=True)
    from tensorflow.python.profiler import traceme
    profile_trace = traceme.TraceMe

if tf.__version__ < TF_REF_VERSION:
    try:
        from tensorflow.python.keras.mixed_precision.experimental import \
            loss_scale_optimizer as lso
    except:
        from tensorflow.python.keras.mixed_precision import \
            loss_scale_optimizer as lso
else:
    from tensorflow.python.keras.mixed_precision import loss_scale_optimizer as lso

from tensorflow.python.distribute import parameter_server_strategy

from ..constants import (KF_UI_META_FILE, AWFUserFunc, DatasetType,
                         DistributionType, RunType)
from ..generators import flatten_seq
from ..utils import (call_user_module, expand_param, make_abs_or_data_path,
                     make_abs_or_pack_path, parse_size, split_file_name)
from .extend_metrics import (CumulationMetric, ExtendedMetric,
                             ExtendedMetricsContainer)
from .extend_utils import (ExtendedLossesContainer, PCGrad,
                           is_using_mixed_precision)
from .helperfuncs import (create_grad_process, create_loss, create_metric,
                          create_optimizer, is_distributed_strategy)


class InputsModifier(object):
    def __init__(self, model_call_exclude_input_index=None, squeeze=False):
        """
        InputsModifier类是对输入x进行一些修改以适配一些特殊的训练流程（例如GAUC计算是需要从x中提取uid信息，
        但是uid又不作为特征进入模型计算）

        Args:
            model_call_exclude_input_index (list/tuple/set, optional): 模型计算时需要从输入中排除掉的特征索引。
            当x是数组时，model_call_exclude_input_index指定的是要排除的序号；
            当x是dict时，model_call_exclude_input_index指定的是要排除的key.
            Defaults to None.

            squeeze (bool, optional): 当从x中排除指定特征后，如果只剩下一个特征，是否去掉最后这个特征的索引/key.
            Defaults to False.
        """
        self.model_call_exclude_input_index = None
        if isinstance(model_call_exclude_input_index, (list, tuple, set)):
            self.model_call_exclude_input_index = set(model_call_exclude_input_index)
        elif model_call_exclude_input_index is not None:
            self.model_call_exclude_input_index = set([model_call_exclude_input_index])
        self.squeeze = squeeze

    def modify_inputs(self, x):
        mod_x = x
        if self.model_call_exclude_input_index:
            if isinstance(x, (tuple, list)):
                mod_x = [x[i] for i in range(len(x)) if i not in self.model_call_exclude_input_index]
                if self.squeeze and len(mod_x) == 1:
                    mod_x = mod_x[0]
            elif isinstance(x, dict):
                mod_x = {k: v for k, v in x.items() if k not in self.model_call_exclude_input_index}

        return mod_x


class ExtendedTrainStep(InputsModifier):
    def __init__(self, model_call_exclude_input_index=None, squeeze=False, multi_optimizer=None, user_module=None,
                 use_pcgrad=None):
        """
        ExtendedTrainStep是对keras model的train_step的改写以支持训练流程的自定义

        Args:
            model_call_exclude_input_index (list/tuple/set, optional): 模型计算时需要从输入中排除掉的特征索引。
            当x是数组时，model_call_exclude_input_index指定的是要排除的序号；
            当x是dict时，model_call_exclude_input_index指定的是要排除的key
            Defaults to None.

            squeeze (bool, optional): 当从x中排除指定特征后，如果只剩下一个特征，是否去掉最后这个特征的索引/key.
            Defaults to False.

            multi_optimizer (bool, optional): 是否使用多优化器. Defaults to None.

            user_module (module, optional): 包含用户代码的python模块. Defaults to None.

            use_pcgrad (bool, optional): 是否使用pcgrad对梯度进行正则化. Defaults to None.
        """
        super(ExtendedTrainStep, self).__init__(model_call_exclude_input_index, squeeze)
        self.multi_optimizer = multi_optimizer
        self.user_module = user_module
        self.use_pcgrad = use_pcgrad

    def __call__(self, model_self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # 对输入x进行必要修改
        mod_x = self.modify_inputs(x)

        # 由于需要兼容使用PCGrad, 因此在获取GradientTape时需要设置persistent为True, 使tape可以被计算多次
        with backprop.GradientTape(persistent=True) as tape:
            y_pred = model_self(mod_x, training=True)
            loss = model_self.compiled_loss(y, y_pred, sample_weight, regularization_losses=model_self.losses,
                                            return_list=self.use_pcgrad)

        self.minimize(model_self, tape, loss)
        del tape # 需要手动清空tape, 否则多次使用tape会积累计算图, 增加内存开销
        model_self.compiled_metrics.update_state(y, y_pred, sample_weight, x)
        return {m.name: m.result() for m in model_self.metrics}

    def minimize(self, model, tape, loss):
        if self.multi_optimizer:
            # 当使用多优化器时，需要用户实现awf_group_trainable_vars_fn回调来对模型trainable参数分组
            trainable_var_groups, _ = call_user_module(self.user_module, AWFUserFunc.GROUP_TRAINABLE_VARS,
                                                       True, False, (list, tuple), model=model)
            self.multi_optimizer_minimize(model.distribute_strategy, model.optimizer, tape, loss, trainable_var_groups)
        else:
            self.single_optimizer_minimize(model.distribute_strategy, model.optimizer, tape, loss,
                                           model.trainable_variables)

    @classmethod
    def __apply_gradients(cls, is_scale_opt, dist_strategy, optimizer, gradients, trainable_variables):
        aggregate_grads_outside_optimizer = (
                optimizer._HAS_AGGREGATE_GRAD and
                not isinstance(dist_strategy.extended,
                               parameter_server_strategy.ParameterServerStrategyExtended))
        if aggregate_grads_outside_optimizer:
            gradients = optimizer._aggregate_gradients(zip(gradients, trainable_variables))

        if is_scale_opt:
            gradients = optimizer.get_unscaled_gradients(gradients)
        gradients = optimizer._clip_gradients(gradients)
        if trainable_variables:
            if aggregate_grads_outside_optimizer:
                optimizer.apply_gradients(
                    zip(gradients, trainable_variables),
                    experimental_aggregate_gradients=False)
            else:
                optimizer.apply_gradients(zip(gradients, trainable_variables))

    @classmethod
    def single_optimizer_minimize(cls, dist_strategy, optimizer, tape, loss, trainable_variables):
        with tape:
            is_scale_opt = isinstance(optimizer, lso.LossScaleOptimizer)
            if is_scale_opt:
                loss = optimizer.get_scaled_loss(loss)
        if isinstance(optimizer, PCGrad):
            # 如果需要使用PCGrad, 那么需要使用多任务的loss对梯度进行预处理, 得到新的梯度
            grads_and_vars = optimizer.process_gradients(loss, trainable_variables, tape)
            gradients = [grad_var[0] for grad_var in grads_and_vars]
        else:
            gradients = tape.gradient(loss, trainable_variables)
        cls.__apply_gradients(is_scale_opt, dist_strategy, optimizer, gradients, trainable_variables)

    @classmethod
    def multi_optimizer_minimize(cls, dist_strategy, optimizers, tape, loss, trainable_var_groups):
        if len(trainable_var_groups) != len(optimizers):
            raise RuntimeError("#optimizers {} != #trainable variable groups {}"
                               .format(len(optimizers), len(trainable_var_groups)))
        is_scale_opt = isinstance(optimizers[0], lso.LossScaleOptimizer)
        gradient_groups = tape.gradient(loss, trainable_var_groups)
        for optimizer, gradients, trainable_variables in zip(optimizers, gradient_groups, trainable_var_groups):
            if isinstance(optimizer, PCGrad):
                # 如果需要使用PCGrad, 那么需要使用多任务的loss对梯度进行预处理, 得到新的梯度
                grads_and_vars = optimizer.process_gradients(loss, trainable_variables, tape)
                gradients = [grad_var[0] for grad_var in grads_and_vars]
            cls.__apply_gradients(is_scale_opt, dist_strategy, optimizer, gradients, trainable_variables)


class ExtendedTestStep(InputsModifier):
    def __init__(self, model_call_exclude_input_index=None, squeeze=True):
        """
        ExtendedTestStep是对keras model的test_step的改写以支持评估流程的自定义

        Args:
            model_call_exclude_input_index (list/tuple/set, optional): 模型计算时需要从输入中排除掉的特征索引。
            当x是数组时，model_call_exclude_input_index指定的是要排除的序号；
            当x是dict时，model_call_exclude_input_index指定的是要排除的key
            Defaults to None.

            squeeze (bool, optional): 当从x中排除指定特征后，如果只剩下一个特征，是否去掉最后这个特征的索引/key.
                Defaults to True.
        """
        super(ExtendedTestStep, self).__init__(model_call_exclude_input_index, squeeze)

    def __call__(self, model_self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        mod_x = self.modify_inputs(x)

        y_pred = model_self(mod_x, training=False)
        model_self.compiled_loss(y, y_pred, sample_weight, regularization_losses=model_self.losses)

        model_self.compiled_metrics.update_state(y, y_pred, sample_weight, x)
        return {m.name: m.result() for m in model_self.metrics}


@enable_multi_worker
def extended_evaluate(model_self,
                      x=None,
                      y=None,
                      batch_size=None,
                      verbose=1,
                      sample_weight=None,
                      steps=None,
                      callbacks=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      return_dict=False):
    """
    对keras model的evaluate函数的的改写以支持评估流程的自定义，参数与流程原生的evaluate函数一致，
    主要是在metric结果的收集上对自定义的metric进行的支持，见下面的注释
    """
    _keras_api_gauge.get_cell('evaluate').set(True)
    version_utils.disallow_legacy_graph('Model', 'evaluate')
    model_self._assert_compile_was_called()
    model_self._check_call_args('evaluate')
    _disallow_inside_tf_function('evaluate')

    with model_self.distribute_strategy.scope():
        if getattr(model_self, '_eval_data_handler', None) is not None:
            data_handler = model_self._eval_data_handler
        else:
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            data_handler = data_adapter.DataHandler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                initial_epoch=0,
                epochs=1,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=model_self,
                steps_per_execution=model_self._steps_per_execution)

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=model_self,
                verbose=verbose,
                epochs=1,
                steps=data_handler.inferred_steps)

        logs = {}
        test_function = model_self.make_test_function()
        model_self._test_counter.assign(0)
        callbacks.on_test_begin()
        for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
            model_self.reset_metrics()
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    with profile_trace('TraceContext', graph_type='test', step_num=step):
                        callbacks.on_test_batch_begin(step)
                        tmp_logs = test_function(iterator)
                        if data_handler.should_sync:
                            context.async_wait()
                        logs = tmp_logs  # No error, now safe to assign to logs.
                        end_step = step + data_handler.step_increment
                        callbacks.on_test_batch_end(end_step, logs)

        # 对自定义CumulationMetric进行适配
        for m in model_self.metrics:
            if isinstance(m, CumulationMetric):
                logs[m.name] = m.final_result()

        logs = tf_utils.to_numpy_or_python_type(logs)
        callbacks.on_test_end(logs=logs)
        if return_dict:
            return logs
        else:
            results = [logs.get(name, None) for name in model_self.metrics_names]
            if len(results) == 1:
                return results[0]
            return results


class TFModelRunner(object):
    def __init__(self, user_py_file, export_path, pack_path, tf_config=None, model_args: dict = None,
                 train_args: dict = None, evaluate_args: dict = None, predict_args: dict = None,
                 train_data_args: dict = None, test_data_args: dict = None, val_data_args: dict = None,
                 predict_data_args: dict = None, save_model_args: dict = None, load_model_args: dict = None):
        """
        TFModelRunner对模型的训练、评估、离线预测流程进行封装，各流程细节处理都通过输入参数进行配置

        Args:
            user_py_file (str): 用户python文件，用户在该文件中进行各回调函数的定义，关于回调函数的定义请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001726069

            export_path (str): 数据目录，关于数据目录定义请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc0

            pack_path (str): 包目录，关于包目录定义请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc0

            tf_config (dict, optional): 分布式训练时的集群节点描述，具体格式请参考tf官方文档. 为None时表示单机.

            model_args (dict, optional): 模型参数，关于模型参数说明请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc11

            train_args (dict, optional): 训练流程参数，关于训练流程参数说明请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc11

            evaluate_args (dict, optional): 评估流程参数，关于评估流程参数请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc13

            predict_args (dict, optional): 离线预测参数. 关于离线预测参数请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc14

            train_data_args (dict, optional): 训练数据集参数，关于训练数据参数请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc11

            test_data_args (dict, optional): 测试数据集参数. 关于测试数据参数请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc13

            val_data_args (dict, optional): 验证数据集参数. 关于验证数据参数请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc13

            predict_data_args (dict, optional): 离线预测数据集参数. 关于离线预测数据参数请参考文档
            http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc14

            save_model_args (dict, optional): 模型保存参数. 传递给awf_model_to_save_fn回调的参数，
            参数内容由用户自定义，关于模型保存回调的说明请查看_model_to_save函数的注释说明

            load_model_args (dict, optional): 模型加载参数，传递给awf_load_model_fn回调的参数，
            参数内容由用户自定义，关于模型加载回调的说明请查看_load_model函数的注释说明
        """
        print("tf version='{}'".format(tf.__version__), flush=True)

        self.user_py_file = make_abs_or_pack_path(user_py_file, export_path, pack_path)
        self.export_path = export_path
        self.pack_path = pack_path
        self.tf_config = tf_config
        self.model_args = model_args or {}
        self.train_data_args = train_data_args or {}
        self.test_data_args = test_data_args or {}
        self.val_data_args = val_data_args or {}
        self.predict_data_args = predict_data_args or {}
        self.train_args = train_args or {}
        self.evaluate_args = evaluate_args or {}
        self.predict_args = predict_args or {}
        self.save_model_args = save_model_args or {}
        self.load_model_args = load_model_args or {}
        user_path = os.path.dirname(os.path.realpath(self.user_py_file))
        user_file_name = os.path.basename(self.user_py_file)
        sys.path.append(user_path)
        self.user_module = __import__(user_file_name.split('.')[0],
                                      fromlist=[AWFUserFunc.CRETAE_MODEL,
                                                AWFUserFunc.CREATE_TRAIN_DATASET,
                                                AWFUserFunc.CREATE_VALIDATION_DATASET,
                                                AWFUserFunc.CREATE_TEST_DATASET,
                                                AWFUserFunc.CREATE_MODEL_TO_SAVE,
                                                AWFUserFunc.LOAD_MODEL])
        print("loaded user python module from file '{}'".format(self.user_py_file))

    def _init_envs(self):
        """
        根据配置设置一些环境变量

        Returns:
            _type_: _description_
        """
        dist_type = self.get_distributed_type()
        envs = {}

        if dist_type == DistributionType.PS:
            # 这个是根据tf的官方文档说明来的，当使用PSStrategy时，需要设置这个环境变量
            envs["GRPC_FAIL_FAST"] = "use_caller"
        elif dist_type == DistributionType.TMEPS:
            # 如果是使用BytePS，则需要设置BytePS相关的环境参数
            # BytePS里面主要有三种角色scheduler，server，worker，
            # 在PS方式下，原生的tf_config中也是三种角色：chief，ps，worker
            # 两者的对应关系为 scheduler <-> chief, server <--> ps，worker <--> worker

            num_workers = self.get_num_workers()
            num_pss = self.get_num_pss()
            role, index = self.get_task_info()
            scheduler_host, scheduler_port = self.get_task_host_info("chief")

            envs = {
                "DMLC_NUM_WORKER": num_workers,
                "DMLC_NUM_SERVER": num_pss,
                "DMLC_PS_ROOT_URI": scheduler_host,
                "DMLC_PS_ROOT_PORT": scheduler_port
            }
            if role != "chief":
                self_host, self_port = self.get_task_host_info()
                envs["DMLC_NODE_HOST"] = self_host
                envs["DMLC_PORT"] = self_port
            if num_workers < 2:
                envs["BYTEPS_FORCE_DISTRIBUTED"] = 1
            if role == 'worker':
                envs["DMLC_ROLE"] = "worker"
                envs["DMLC_WORKER_ID"] = index
                envs["BYTEPS_LOCAL_RANK"] = 0
                envs["BYTEPS_LOCAL_SIZE"] = 1
            elif role == 'chief':
                envs["DMLC_ROLE"] = "scheduler"
            elif role == 'ps':
                envs["DMLC_ROLE"] = "server"

            if self.train_args.get('debug', False):
                envs["BYTEPS_SERVER_DEBUG"] = 1
                envs["NCCL_DEBUG"] = "WARN"
            ps_config = self.tf_config.get('ps_config', {})
            for k, v in ps_config.items():
                # 下面在为了调试时，可以增加一些日志输出
                if k == 'verbose':
                    envs['PS_VERBOSE'] = v
                elif k == 'log_level':
                    envs['BYTEPS_LOG_LEVEL'] = v
                elif k == 'log_key':
                    envs['PS_KEY_LOG'] = v
                elif k == 'debug':
                    envs['BYTEPS_SERVER_DEBUG'] = v
                elif k == 'debug_key':
                    envs['BYTEPS_SERVER_DEBUG_KEY'] = v

        if envs:
            print("set environment variables for '{}': {}".format(dist_type, envs))
            for k, v in envs.items():
                os.environ[k] = str(v)

        return envs

    def is_chief(self):
        dist_type = self.get_distributed_type()
        if dist_type in [DistributionType.NONE, DistributionType.SINGLE_WORKER]:
            return True

        role, index = self.get_task_info()
        if role:
            role = role.strip().lower()

        cluster = self.get_cluster_info()

        if dist_type != DistributionType.TMEPS:
            if role in ['chief', 'master']:
                return True
            if 'chief' in cluster or 'master' in cluster:
                return False
        return role == 'worker' and index == 0

    def get_num_workers(self):
        if not self.tf_config:
            return 1
        cluster = self.get_cluster_info()
        workers = cluster.get('worker', [])
        return len(workers)

    def get_num_pss(self):
        dist_type = self.get_distributed_type()
        if dist_type not in [DistributionType.TMEPS, DistributionType.PS]:
            return 0
        cluster = self.get_cluster_info()
        pss = cluster.get('ps', [])
        return len(pss)

    def get_distributed_type(self):
        """
        根据tf_config返回当前的分布式方式

        Returns:
            DistributionType: 分布式方式
        """
        if not self.tf_config:
            return DistributionType.NONE
        cluster_info = self.tf_config.get('cluster', {})
        if not cluster_info:
            return DistributionType.NONE

        workers = cluster_info.get('worker', [])
        pss = cluster_info.get('ps', [])
        chiefs = cluster_info.get('chief', [])
        masters = cluster_info.get('master', [])
        if len(pss) > 0:
            # 这里对tf_config做了个扩展，可以包含ps_config用来指定ps的框架，配置等信息
            ps_config = self.tf_config.get('ps_config', {})
            framework = ps_config.get('framework')
            if framework == 'tme':
                if len(workers) > 1:
                    return DistributionType.TMEPS
                else:
                    # BytePS在worker数为1的时候并不会启用分布式，会回退到单机模式
                    print("currently TEMPS only works with multiple workers, fallback to normal "
                          "single-worker model when only 1 worker specified")
                    return DistributionType.SINGLE_WORKER
            return DistributionType.PS

        if len(workers) + len(chiefs) + len(masters) > 1:
            return DistributionType.MULTI_WORKER

        return DistributionType.SINGLE_WORKER

    def get_task_info(self):
        if not self.tf_config:
            return None, None
        task_info = self.tf_config.get('task', {})
        if not task_info:
            return None, None
        return task_info.get('type'), task_info.get('index')

    def get_task_host_info(self, task_type=None, task_index=0):
        if not self.tf_config:
            return None, None
        if not task_type:
            task_type, task_index = self.get_task_info()
        if not task_type:
            return None, None
        tasks = self.tf_config.get("cluster", {}).get(task_type, [])
        if len(tasks) <= task_index:
            return None, None
        task_host = tasks[task_index]
        host, port = task_host.split(":")
        return host, port

    def get_cluster_info(self):
        if not self.tf_config:
            return None
        return self.tf_config.get('cluster')

    def _create_model(self) -> tf.keras.Model:
        """
        调用用户回调创建模型，用户回调中需要模型定义并返回Model对象

        Returns:
            tf.keras.Model: 创建的模型实例
        """

        continue_training = self.model_args.get("continue_training", False)
        continue_training_model_path = self.model_args.get("continue_training_model_path", "")
        if continue_training and not os.path.exists(continue_training_model_path):
            continue_training = False  # 首次增量训练从头开始训练
            print("continue_training_model_path '{}' is not exists, so change to normal training process"
                  .format(continue_training_model_path))

        inject_args = {'pack_path': self.pack_path, 'data_path': self.export_path, 'export_path': self.export_path}
        if continue_training:
            # 对于增量训练需要指定base模型路径，然后调用awf_load_model_fn回调加载base模型
            continue_training_model_name = self.model_args.get("name", "default")
            load_model_args = {'path': continue_training_model_path, 'name': continue_training_model_name}
            model, _ = call_user_module(self.user_module, AWFUserFunc.LOAD_MODEL, False, True, tf.keras.Model,
                                        inject_args=inject_args, **load_model_args)
            print("loaded model from '{}' for continue training".format(continue_training_model_path))
            name = self.model_args.get('name', '').strip()
            if name and model.name != name:
                orig_name = model.name
                model._name = name
                print("changed loaded model('{}') name '{}' to '{}'".format(continue_training_model_path,
                                                                            orig_name, model.name))
        else:
            model, _ = call_user_module(self.user_module, AWFUserFunc.CRETAE_MODEL, True, False, tf.keras.Model,
                                        inject_args=inject_args, **self.model_args)
            print("created model for fresh start training")
        return model

    def _create_dateset(self, dataset_type, dataset_args, repeat, global_batch_size, drop_remainder,
                        shuffle_buffer_size=None, num_shards=None, shard_index=None,
                        check_return_type=tf.data.Dataset) -> Union[tf.data.Dataset, list, dict, tuple]:
        """
        调用用户回调创建数据集，用户回调中需要包含数据解析逻辑并返回Dataset对象

        Args:
            dataset_type (DatasetType): 数据集类型
            dataset_args (dict): 传递给用户回调的参数，可以用户自定义
            repeat (bool): 是否对数据集进行repeat（多机分布式下为true）
            global_batch_size (int): 全局batch大小
            drop_remainder (bool): 是否丢掉数据集最后不足一个batch的尾部（多机分布式下为true）
            shuffle_buffer_size (int, optional): 如果要shuffle数据的话，shuffle的buffer大小. Defaults to None.
            num_shards (int, optional): 数据分片数（只针对ps方式）. Defaults to None.
            shard_index (int, optional): 当前worker对应分片的index（只针对ps方式）. Defaults to None.
            check_return_type (optional): 限定用户回调返回值的类型. Defaults to tf.data.Dataset.

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_

        Returns:
            Union[tf.data.Dataset, list, dict, tuple]: _description_
        """

        if dataset_type == DatasetType.TRAIN:
            func_name = AWFUserFunc.CREATE_TRAIN_DATASET
        elif dataset_type == DatasetType.TEST:
            func_name = AWFUserFunc.CREATE_TEST_DATASET
        elif dataset_type == DatasetType.VALIDATION:
            func_name = AWFUserFunc.CREATE_VALIDATION_DATASET
        elif dataset_type == DatasetType.PREDICTION:
            func_name = AWFUserFunc.CREATE_PREDICTION_DATASET
        else:
            raise RuntimeError("unknown dataset type '{}'".format(dataset_type))

        args = dataset_args.copy()
        for k, v in args.items():
            if isinstance(v, str):
                args[k] = expand_param(v, self.export_path, self.pack_path)

        shard_policy = args.get('shard_policy', 'AUTO').strip().upper()
        if shard_policy == 'DATA':
            shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        elif shard_policy == 'FILE':
            shard_policy = tf.data.experimental.AutoShardPolicy.FILE
        else:
            shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

        dataset, injected_args = call_user_module(self.user_module, func_name, False, True, check_return_type,
                                                  inject_args={"batch_size": global_batch_size,
                                                               "shard_policy": shard_policy,
                                                               "is_chief": self.is_chief(),
                                                               "repeat": repeat,
                                                               "drop_remainder": drop_remainder,
                                                               "pack_path": self.pack_path,
                                                               "data_path": self.export_path,
                                                               "export_path": self.export_path},
                                                  **args)
        if dataset is None:
            print("user function '{}' return None dataset, args={}".format(func_name, args))
            return dataset

        def __apply_options(ds):
            """
            根据参数对Dataset应用所需的选项
            """
            if shuffle_buffer_size is not None and shuffle_buffer_size > 0:
                ds = ds.shuffle(shuffle_buffer_size)
                print("shuffled {} dataset with buffer_size {}".format(dataset_type, shuffle_buffer_size))

            if 'repeat' in injected_args:
                print("injected 'repeat' in user function '{}', will not do repeat".format(func_name))
            elif repeat:
                print("user function '{}' has no 'repeat' arg, repeated dataset".format(func_name))
                ds = ds.repeat()

            sharded_dataset = False
            if num_shards is not None and num_shards > 0 and shard_index is not None and shard_index >= 0:
                ds = ds.shard(num_shards, shard_index)
                sharded_dataset = True
                print("sharded {} dataset with {} shards, shard_index={}".format(dataset_type, num_shards, shard_index))

            if 'batch_size' in injected_args:
                print("injected 'batch_size' in user function '{}', will not do batch".format(func_name))
            else:
                print("user function '{}' has no 'batch_size' arg, applied batch {} to dataset"
                      .format(func_name, global_batch_size))
                ds = ds.batch(global_batch_size, drop_remainder)
            if not sharded_dataset:
                options = tf.data.Options()
                options.experimental_distribute.auto_shard_policy = shard_policy
                print("set {} dataset shard policy to '{}'".format(dataset_type, shard_policy))
                ds = ds.with_options(options)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return ds

        if isinstance(dataset, tf.data.Dataset):
            return __apply_options(dataset)
        elif isinstance(dataset, (tuple, list)):
            return [__apply_options(ds) for ds in dataset]
        elif isinstance(dataset, dict):
            return {k: __apply_options(ds) for k, ds in dataset.items()}
        else:
            raise RuntimeError("user function '{}' return unsupported dataset type, args={}, only tf.data.Dataset or"
                               " list/tuple/dict of tf.data.Dataset are supported, got type '{}': {}"
                               .format(func_name, args, type(dataset), dataset))

    def _create_optimizer(self, loss_scaled=False, v1=False, **kwargs):
        """
        根据用户配置创建优化器

        Args:
            loss_scaled (bool, optional): 是否使用loss缩放，在使用混合精度时为true. Defaults to False.
            v1 (bool, optional): 是否使用tf.compate.v1中的优化器，这是为了适配estimator. Defaults to False.
        """
        def __create_one_optimizer(optimizer_detail):
            """
            根据配置，创建单个优化器

            Args:
                optimizer_detail (dict/str): 优化器配置
            Raises:
                RuntimeError: _description_
                NotImplementedError: _description_

            Returns:
                tf.optimizers.Optimizer: 创建的优化器
            """
            args = {}
            if not optimizer_detail:
                optimizer_type = 'adam'
                lr = 0.001
                print("'optimizer' not set, type default to '{}' and learning rate default to {}"
                      .format(optimizer_type, lr))
            else:
                if not isinstance(optimizer_detail, (str, dict)):
                    raise RuntimeError("optimizer must be str or dict, got '{}': {}"
                                       .format(type(optimizer_detail), optimizer_detail))

                if isinstance(optimizer_detail, str):
                    optimizer_type = optimizer_detail.strip() or 'adam'
                    lr = 0.001
                    print("'learning rate not set, default to {}'".format(lr))
                else:
                    optimizer_type = optimizer_detail.get('type')
                    if not optimizer_type or not optimizer_type.strip():
                        optimizer_type = 'adam'
                        lr = 0.001
                        print("type of optimizer not set, type default to '{}' and learning rate default to {}"
                              .format(optimizer_type, lr))
                    else:
                        optimizer_type = optimizer_type.strip()
                        args = copy.deepcopy(optimizer_detail.get('args', {}))
                        if not args or ('learning_rate' not in args and 'lr' not in args):
                            lr = 0.001
                            print("'learning rate not set, default to {}'".format(lr))
                        else:
                            lr = args.get('learning_rate') or args.get('lr')
                            if 'lr' in args:
                                args.pop('lr')
                            if 'learning_rate' in args:
                                args.pop('learning_rate')

            if args and kwargs:
                args.update(kwargs)
            elif kwargs:
                args = kwargs
            elif not args:
                args = {}
            optimizer = create_optimizer(optimizer_type, lr, v1, **args)
            if optimizer is None:
                raise NotImplementedError("unsupported optimizer type '{}', v1={}".format(optimizer_type, v1))

            if loss_scaled:
                if tf.__version__ < TF_REF_VERSION:
                    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer,
                                                                                         loss_scale='dynamic')
                else:
                    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                print("wrapped optimizer '{}' with LossScaleOptimizer".format(optimizer_type))

            if optimizer_detail is not None and isinstance(optimizer_detail, dict):
                grad_process = optimizer_detail.get('grad_process')
                grad_process = create_grad_process(grad_process)
                if grad_process is not None:
                    optimizer = grad_process(optimizer)
            print("create optimizer of type '{}' of learning rate {}, args={}: {}"
                  .format(optimizer_type, lr, args, optimizer))
            return optimizer

        optimizer_specs = self.train_args.get('optimizer')
        if isinstance(optimizer_specs, (list, tuple)):
            if len(optimizer_specs) <= 1:
                return __create_one_optimizer(None) if not optimizer_specs else \
                    __create_one_optimizer(optimizer_specs[0])
            return [__create_one_optimizer(spec) for spec in optimizer_specs]
        return __create_one_optimizer(optimizer_specs)

    def _create_losses(self, model, is_training=True, **kwargs):
        """
        根据用户配置创建loss，除了内置的loss，也支持用户自定义loss

        Args:
            model (tf.keras.Model): 模型实例
            is_training (bool, optional): 是否是用于训练（否则就是用于评估）. Defaults to True.

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
            RuntimeError: _description_
            NotImplementedError: _description_
            RuntimeError: _description_
            NotImplementedError: _description_
            RuntimeError: _description_

        Returns:
            tf.losses.Loss: 创建的损失对象
        """
        if is_training:
            losses_detail = self.train_args.get('losses')
        else:
            losses_detail = self.evaluate_args.get('losses') or self.train_args.get('losses')

        if not losses_detail:
            if is_training:
                raise RuntimeError("losses not set, is_training={}".format(is_training))
            else:
                return None, None
        if not isinstance(losses_detail, (str, dict, list)):
            raise RuntimeError("losses must be str or dict or list, got {}, is_training={}"
                               .format(losses_detail, is_training))

        inject_args = {'pack_path': self.pack_path, 'data_path': self.export_path, 'export_path': self.export_path}

        def __parse_single_loss(detail):
            """
            根据配置创建单个loss

            Args:
                detail (str/dict): 损失配置

            Raises:
                RuntimeError: _description_
                NotImplementedError: _description_
                RuntimeError: _description_
                NotImplementedError: _description_

            Returns:
                tf.losses.Loss: 创建的损失对象
            """
            if isinstance(detail, str):
                loss_type = detail.strip()
                if not loss_type:
                    raise RuntimeError("loss can not be empty string, is_training={}".format(is_training))
                args = kwargs or {}
                loss = create_loss(loss_type, **args)
                if loss is None:
                    # 尝试调用用户函数创建用户自定义损失，函数名就是配置的loss类型，返回一个Loss对象
                    loss, _ = call_user_module(self.user_module, loss_type, False, False,
                                               tf.keras.losses.Loss, inject_args=inject_args, model=model)
                    if loss is not None:
                        print("created loss {} from user function '{}'".format(loss, loss_type))
                    else:
                        raise NotImplementedError("unsupported loss type '{}', is_training={}"
                                                  .format(loss_type, is_training))
                print("create '{}' loss {} by str '{}', is_training={}".format(loss_type, loss, detail, is_training))
                return loss, None
            else:
                loss_type = detail.get('type')
                if not loss_type or not loss_type.strip():
                    raise RuntimeError("loss type not set, is_training={}".format(is_training))
                loss_type = loss_type.strip()
                args = copy.deepcopy(detail.get('args', {}))
                if kwargs:
                    args.update(kwargs)
                loss = create_loss(loss_type, **args)
                if loss is None:
                    ud_args = detail.get('args', {})
                    ud_args['model'] = model
                    loss, _ = call_user_module(self.user_module, loss_type, False, False,
                                               tf.keras.losses.Loss, inject_args=inject_args, **ud_args)
                    if loss is not None:
                        print("created loss {} from user function '{}', args={}".format(loss, loss_type, ud_args))
                    else:
                        raise NotImplementedError("unsupported loss type '{}', is_training={}"
                                                  .format(loss_type, is_training))
                loss_weight = detail.get('weight')
                if loss_weight is not None:
                    loss_weight = float(loss_weight)
                print("create '{}' loss {} by dict '{}' with weight {}, is_training={}"
                      .format(loss_type, loss, detail, loss_weight, is_training))
                return loss, loss_weight

        if isinstance(losses_detail, (str, dict)):
            l, _ = __parse_single_loss(losses_detail)
            return l, None

        losses = []
        weights = []
        for i, loss_item in enumerate(losses_detail):
            if not isinstance(loss_item, (str, dict)):
                raise RuntimeError("{}th loss must be str or dict, got {}, is_training={}"
                                   .format(i, loss_item, is_training))
            l, w = __parse_single_loss(loss_item)
            losses.append(l)
            weights.append(w)
        if len(losses) > 1 and not all(weights):
            print("not all loss weights are valid: {}, will set all weights to 1, is_training={}"
                  .format(weights, is_training))
            weights = [1.] * len(losses)
        elif len(losses) == 1:
            losses = losses[0]
            weights = None
        return losses, weights

    def _create_metrics(self, model, name_prefix=None, is_training=True, v1=False, **kwargs):
        """
        根据用户配置创建metrics，除了内置的metric，也支持用户自定义metric

        Args:
            model (tf.keras.Model): 模型实例
            name_prefix (str, optional): metric的命名前缀. Defaults to None.
            is_training (bool, optional): 是否是用于训练（否则就是用于评估）. Defaults to True.
            v1 (bool, optional): 是否使用tf.compat.v1.metrics中的metric，这是为了适配estimator. Defaults to False.

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
            NotImplementedError: _description_
            RuntimeError: _description_
            NotImplementedError: _description_
            RuntimeError: _description_
            RuntimeError: _description_
            RuntimeError: _description_
            RuntimeError: _description_
            RuntimeError: _description_
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
        if is_training:
            metrics_detail = self.train_args.get('metrics')
        else:
            metrics_detail = self.evaluate_args.get('metrics') or self.train_args.get('metrics')

        if not metrics_detail:
            print("metrics not set, is_training={}".format(is_training))
            return None, None
        if not isinstance(metrics_detail, (str, dict, list)):
            raise RuntimeError("metrics must be str or dict or list, got {}, is_training={}"
                               .format(metrics_detail, is_training))

        inject_args = {'pack_path': self.pack_path, 'data_path': self.export_path, 'export_path': self.export_path}

        def __parse_single_metric(detail, output_idx=None):
            if isinstance(detail, str):
                metric_type = detail.strip()
                if not metrics_detail:
                    raise RuntimeError("metric can not be empty string, is_training={}".format(is_training))
                args = kwargs or {}
                if output_idx is not None:
                    args['name'] = args.get('name', metric_type).strip() + '_' + str(output_idx)
                metric = create_metric(metric_type, name_prefix, v1=v1, **args)
                if metric is None:
                    # 尝试调用用户函数创建用户自定义metric，函数名就是配置的metric类型，返回一个metric对象
                    metric, _ = call_user_module(self.user_module, metric_type, False, False,
                                                 tf.keras.metrics.Metric, inject_args=inject_args, model=model)
                    if metric is not None:
                        print("created metric {} from user function '{}'".format(metric, metric_type))
                    else:
                        raise NotImplementedError("unsupported metic type '{}', is_training={}"
                                                  .format(metric_type, is_training))
                print("create '{}' '{}' metric {} by str '{}', is_training={}"
                      .format(name_prefix, metric_type, metric, detail, is_training))
            else:
                metric_type = detail.get('type')
                if not metric_type or not metric_type.strip():
                    raise RuntimeError("metric type not set, is_training={}".format(is_training))
                metric_type = metric_type.strip()
                args = copy.deepcopy(detail.get('args', {}))
                if kwargs:
                    args.update(kwargs)
                if output_idx is not None:
                    args['name'] = args.get('name', metric_type).strip() + '_' + str(output_idx)
                metric = create_metric(metric_type, name_prefix, v1=v1, **args)
                if metric is None:
                    ud_args = detail.get('args', {})
                    ud_args['model'] = model
                    if output_idx is not None:
                        ud_args['name'] = ud_args.get('name', metric_type).strip() + '_' + str(output_idx)
                    metric, _ = call_user_module(self.user_module, metric_type, False, False,
                                                 tf.keras.metrics.Metric, inject_args=inject_args, **ud_args)
                    if metric is not None:
                        print("created metric {} from user function '{}', args={}".format(metric, metric_type, ud_args))
                    else:
                        raise NotImplementedError("unsupported metric type '{}', is_training={}"
                                                  .format(metric_type, is_training))
                print("create '{}' '{}' metric {} by dict '{}', is_training={}"
                      .format(name_prefix, metric_type, metric, detail, is_training))
            return metric_type, metric

        metric_list = []
        metric_dict = {}
        if isinstance(metrics_detail, str) or (isinstance(metrics_detail, dict) and 'type' in metrics_detail):
            metric_type, metric = __parse_single_metric(metrics_detail)
            if v1:
                metric_dict[metric_type] = metric
            else:
                metric_list.append(metric)
        elif isinstance(metrics_detail, dict):
            for output_name, metric_item in metrics_detail.items():
                if not isinstance(metric_item, (str, dict, list)):
                    raise RuntimeError("metric of output '{}' must be str or dict or list, got {}, is_training={}"
                                       .format(output_name, metric_item, is_training))
                if isinstance(metric_item, (str, dict)):
                    metric_type, metric = __parse_single_metric(metric_item)
                    if v1:
                        metric_dict[output_name+'_'+metric_type] = metric
                    else:
                        metric_dict[output_name] = metric
                else:
                    if v1:
                        for j, sub_item in enumerate(metric_item):
                            if not isinstance(sub_item, (str, dict)):
                                raise RuntimeError("({}th metric of output '{}' must be str or dict, got {}, \
                                                   is_training={}".format(j, output_name, sub_item, is_training))
                            sub_metric_type, sub_metric = __parse_single_metric(sub_item)
                            metric_dict[output_name+'_'+sub_metric_type] = sub_metric
                    else:
                        sub_metric_list = []
                        for j, sub_item in enumerate(metric_item):
                            if not isinstance(sub_item, (str, dict)):
                                raise RuntimeError("({}th metric of output '{}' must be str or dict, got {}, \
                                                   is_training={}".format(j, output_name, sub_item, is_training))
                            _, sub_metric = __parse_single_metric(sub_item)
                            sub_metric_list.append(sub_metric)
                        metric_dict[output_name] = sub_metric_list
        else:
            multi_output = len(metrics_detail) > 1
            for i, metric_item in enumerate(metrics_detail):
                if not isinstance(metric_item, (str, dict, list)):
                    raise RuntimeError("{}th metric must be str or dict or list, got {}, is_training={}"
                                       .format(i, metric_item, is_training))
                index = i if multi_output else None
                if isinstance(metric_item, (str, dict)):
                    metric_type, metric = __parse_single_metric(metric_item, index)
                    if v1:
                        metric_dict[metric_type+'_'+str(i) if multi_output else metric_type] = metric
                    else:
                        metric_list.append(metric)
                else:
                    if v1:
                        for j, sub_item in enumerate(metric_item):
                            if not isinstance(sub_item, (str, dict)):
                                raise RuntimeError("({}, {})th metric must be str or dict, got {}, is_training={}"
                                                   .format(i, j, sub_item, is_training))
                            sub_metric_type, sub_metric = __parse_single_metric(sub_item, index)
                            metric_dict[sub_metric_type+'_'+str(i) if multi_output else sub_metric_type] = sub_metric
                    else:
                        sub_metric_list = []
                        for j, sub_item in enumerate(metric_item):
                            if not isinstance(sub_item, (str, dict)):
                                raise RuntimeError("({}, {})th metric must be str or dict, got {}, is_training={}"
                                                   .format(i, j, sub_item, is_training))
                            _, sub_metric = __parse_single_metric(sub_item, index)
                            sub_metric_list.append(sub_metric)
                        metric_list.append(sub_metric_list)

        print("created metrics={}, name_prefix='{}', is_training={}".format(metric_list or metric_dict, name_prefix,
                                                                            is_training))
        return metric_list or metric_dict

    def _create_callbacks(self):
        """
        根据配置创建训练callback

        Returns:
            list: callback列表
        """
        callbacks = []
        ckpt_path = self.train_args.get('ckpt_path')
        if not ckpt_path or not ckpt_path.strip():
            ckpt_path = 'checkpoints'
        else:
            ckpt_path = ckpt_path.strip()
        ckpt_path = make_abs_or_data_path(ckpt_path, self.export_path, self.pack_path)
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
            print("created check point dir '{}'".format(ckpt_path))
        if self.get_distributed_type() != DistributionType.TMEPS and tf.executing_eagerly():
            if tf.__version__ < TF_REF_VERSION:
                from .extend_callbacks import BackupAndStockCallBack
                callbacks.append(BackupAndStockCallBack(ckpt_path))
                print("added BackupAndStockCallBack callback, backup_dir='{}'".format(ckpt_path))
            else:
                callbacks.append(tf.keras.callbacks.experimental.BackupAndRestore(ckpt_path))
                print("added BackupAndRestore callback, backup_dir='{}'".format(ckpt_path))

        early_stop_spec = self.train_args.get("early_stopping")
        if early_stop_spec is not None:
            early_stop_spec['restore_best_weights'] = True
            callbacks.append(tf.keras.callbacks.EarlyStopping(**early_stop_spec))
            print("added early stopping callback, spec={}".format(early_stop_spec))

        if self.get_distributed_type() != DistributionType.PS:
            trainspeed_log_spec = self.train_args.get("train_speed_logger")
            if trainspeed_log_spec is not None:
                from .extend_callbacks import TrainSpeedLoggerCallback
                callbacks.append(TrainSpeedLoggerCallback(**trainspeed_log_spec))
                print("added train speed logger callback, spec={}".format(trainspeed_log_spec))
        else:
            print("under PS distribution, TrainSpeedLoggerCallback callback not supported")

        term_on_nan = self.train_args.get("term_on_nan")
        if term_on_nan:
            callbacks.append(tf.keras.callbacks.TerminateOnNaN())
            print("added terminate on nan callback")

        # 判断是否使用NNI超参搜索, 并进行callback设置
        nni_search = self.train_args.get('nni_search', False)
        nni_search = self.is_chief() and nni_search  # only chief node need to report result
        nni_metric = self.train_args.get("nni_metric")
        nni_batch = self.train_args.get('nni_batch', None)
        nni_exp_id = self.train_args.get("nni_exp_id")
        nni_trial_id = self.train_args.get("nni_trial_id")
        nni_record = self.train_args.get('nni_record')
        print('nni setting', nni_search, 'nni metric', nni_metric, 'nni batch', nni_batch,
              'nni exp_id', nni_exp_id, 'nni_trial_id', nni_trial_id, 'nni_record', nni_record)
        if nni_search:
            from .extend_callbacks import NNISearchPushCallBack
            callbacks.append(NNISearchPushCallBack(nni_exp_id, nni_trial_id, nni_record, nni_batch, nni_metric))
            print("added nni search push callback")

        tensorboard_spec = self.train_args.get('tensorboard')
        if tensorboard_spec is not None:
            summary_path = tensorboard_spec.get('log_dir')
            if not summary_path or not summary_path.strip():
                summary_path = 'summary'
            else:
                summary_path = summary_path.strip()
            summary_path = make_abs_or_data_path(summary_path, self.export_path, self.pack_path)
            with open(KF_UI_META_FILE, 'w') as f:
                ui_meta = {
                    "outputs": [
                        {
                            "type": "tensorboard",
                            "source": summary_path
                        }
                    ]
                }
                json.dump(ui_meta, f)
                print("wrote ui meta data '{}' into '{}'".format(ui_meta, KF_UI_META_FILE))

            if self.is_chief():
                if not os.path.isdir(summary_path):
                    os.makedirs(summary_path, exist_ok=True)
                    print("created summary dir '{}'".format(summary_path))
                tensorboard_spec['log_dir'] = summary_path
                if 'histogram_freq' not in tensorboard_spec:
                    tensorboard_spec['histogram_freq'] = 1
                if 'update_freq' not in tensorboard_spec:
                    tensorboard_spec['update_freq'] = 'batch'
                if 'profile_batch' not in tensorboard_spec:
                    tensorboard_spec['profile_batch'] = 0
                if 'embeddings_freq' not in tensorboard_spec:
                    tensorboard_spec['embeddings_freq'] = 0
                if 'write_graph' not in tensorboard_spec:
                    tensorboard_spec['write_graph'] = True
                if 'write_grads' not in tensorboard_spec:
                    tensorboard_spec['write_grads'] = True
                if tf.__version__ < TF_REF_VERSION:
                    callbacks.append(tf.compat.v1.keras.callbacks.TensorBoard(**tensorboard_spec))
                    print("tf.version=={}, use v1 tensorboard callback".format(tf.__version__))
                else:
                    callbacks.append(tf.keras.callbacks.TensorBoard(**tensorboard_spec))
                    print("tf.version=={}, use v2 tensorboard callback".format(tf.__version__))
                print("added tensorboard callback, spec={}".format(tensorboard_spec))

        return callbacks

    def _model_to_save(self, model: tf.keras.Model) -> list:
        """
        调用用户回调获得需要保存的模型对象及签名，对于有些模型训练逻辑和serving逻辑不同，
        serving部分可能是训练模型的一个子部分，用户可以在回调中定义要保存的模型结构，
        如果用户没有实现相应回调，则默认保存传入的model对象
        用户回调可以返回单个或多个要保存的模型，对于每一个模型，可以单独的一个tf.keras.Model对象，
        也可以是一个tuple，tuple格式为(model, signature, options)，其中signature是模型serving
        接口的签名，options是saved model相关选项（可以参考tf官方文档），signature和options都是
        可选的

        Args:
            model (tf.keras.Model): 训练好的模型对象

        Raises:
            RuntimeError: _description_

        Returns:
            list: 返回要保存的模型列表
        """
        args = self.save_model_args or {}
        args.update({'trained_model': model})
        inject_args = {'pack_path': self.pack_path, 'data_path': self.export_path, 'export_path': self.export_path}
        model_to_save, _ = call_user_module(self.user_module, AWFUserFunc.CREATE_MODEL_TO_SAVE, False, True,
                                            (tf.keras.Model, tuple, list), inject_args=inject_args, **args)

        if not isinstance(model_to_save, list):
            model_to_save = [model_to_save]

        model_save_infos = []
        for item in model_to_save:
            signatures = None
            save_options = None
            if item is not None:
                if isinstance(item, tuple):
                    if len(item) == 0:
                        print("user function '{}' return no model info".format(AWFUserFunc.CREATE_MODEL_TO_SAVE))
                        final_model = model
                    elif len(item) > 3:
                        raise RuntimeError("user function '{}' should return (<Model>, <Signature>, <SaveOption>) or"
                                           " (<Model>, <Signature>) or <Model>, got {}"
                                           .format(AWFUserFunc.CREATE_MODEL_TO_SAVE, item))
                    else:
                        final_model = item[0]
                        if len(item) > 1:
                            signatures = item[1]
                        if len(item) > 2:
                            save_options = item[2]
                else:
                    final_model = item
            else:
                final_model = model
            model_save_infos.append((final_model, signatures, save_options))

        return model_save_infos

    def _patch_model(self, model, metrics, losses, loss_weights, model_call_exclude_input_index=None, squeeze=False,
                     multi_optimizer=False, use_pcgrad=False):
        """
        根据训练流程的配置（是否有特殊metric，是否使用多优化器，是否使用PCGrad等）来对keras model的train_step/test_step
        等流程进行改写

        Args:
            model (tf.keras.Model): 要改写流程的model实例

            metrics (list/dict): 使用的metric集合

            losses (list/dict): 使用的loss集合

            loss_weights (list/dict): loss权重集合

            model_call_exclude_input_index (list/tuple/set, optional): 模型计算时需要从输入中排除掉的特征索引。
            当x是数组时，model_call_exclude_input_index指定的是要排除的序号；
            当x是dict时，model_call_exclude_input_index指定的是要排除的key.
            Defaults to None.

            squeeze (bool, optional): 当从x中排除指定特征后，如果只剩下一个特征，是否去掉最后这个特征的索引/key.
            Defaults to False.

            multi_optimizer (bool, optional): 是否使用多优化器. Defaults to False.

            use_pcgrad (bool, optional): 是否使用pcgrad. Defaults to False.
        """
        if (not metrics or not any(isinstance(m, ExtendedMetric) for m in flatten_seq(metrics))) \
                and model_call_exclude_input_index is None and not multi_optimizer and not use_pcgrad:
            print("no need to patch model '{}' {}, metrics:{}, model_call_exclude_input_index={}, squeeze={}, "
                  "multi_optimizer={}, use_pcgrad={}"
                  .format(model.name, model, metrics, model_call_exclude_input_index, squeeze,
                          multi_optimizer, use_pcgrad))
            return
        metric_container = ExtendedMetricsContainer(metrics, output_names=model.output_names)
        model.compiled_metrics = metric_container
        loss_container = ExtendedLossesContainer(losses, loss_weights, output_names=model.output_names)
        model.compiled_loss = loss_container
        model.train_step = types.MethodType(ExtendedTrainStep(model_call_exclude_input_index, squeeze,
                                                              multi_optimizer, self.user_module, use_pcgrad), model)
        model.test_step = types.MethodType(ExtendedTestStep(model_call_exclude_input_index, squeeze), model)
        model.evaluate = types.MethodType(extended_evaluate, model)
        print("patched model '{}' {} with metrics:{}, model_call_exclude_input_index={}, squeeze={}"
              .format(model.name, model, metrics, model_call_exclude_input_index, squeeze))

    def _train_compile_fit(self, dist_strategy: tf.distribute.Strategy, global_batch_size, steps_per_epoch,
                           validation_steps):
        """
        启动compile_fit训练流程

        Args:
            dist_strategy (tf.distribute.Strategy): 使用的分布式Strategy
            global_batch_size (int): 全局batch大小
            steps_per_epoch (int): 每个epoch的step数
            validation_steps (int): 每个epoch的验证step数

        Returns:
            tf.keras.Model: 训练好的模型实例
        """

        is_distributed = is_distributed_strategy(dist_strategy)

        train_ds = self._create_dateset(DatasetType.TRAIN, self.train_data_args, is_distributed,
                                        global_batch_size, is_distributed)
        if train_ds is None:
            print("failed to load training data, exit, train_data_args={}".format(self.train_data_args))
            return None

        val_ds = self._create_dateset(DatasetType.VALIDATION, self.val_data_args, is_distributed,
                                      global_batch_size, is_distributed)

        if self.get_distributed_type() == DistributionType.PS:
            # TF PS方式下，数据集的传递方式不太一样，需要定义相应的数据集构建函数，然后使用DatasetCreator，
            # 详情可以参考TF的官方文档
            def train_dataset_fn(input_context):
                batch_size = input_context.get_per_replica_batch_size(global_batch_size)
                ds = self._create_dateset(DatasetType.TRAIN, self.train_data_args, True, batch_size, True,
                                          2 * global_batch_size, input_context.num_input_pipelines,
                                          input_context.input_pipeline_id)
                return ds

            def val_dataset_fn(input_context):
                batch_size = input_context.get_per_replica_batch_size(global_batch_size)
                ds = self._create_dateset(DatasetType.VALIDATION, self.val_data_args, True, batch_size, True,
                                          2 * global_batch_size, input_context.num_input_pipelines,
                                          input_context.input_pipeline_id)
                return ds

            train_ds = tf.keras.utils.experimental.DatasetCreator(train_dataset_fn)

            if val_ds is not None:
                val_ds = tf.keras.utils.experimental.DatasetCreator(val_dataset_fn)

            steps_per_execution = self.train_args.get('steps_per_execution', 10)
            print("set steps_per_execution={} under ps training mode".format(steps_per_execution))
        elif self.get_distributed_type() == DistributionType.TMEPS:
            # BytePS的Strategry里面没有自动对数据集做分布式切分，所以这里需要手动的做一下Dataset分布式，否则
            # 所有worker都会遍历所有整个数据集
            num_workers = self.get_num_workers()
            _, worker_idx = self.get_task_info()

            def train_dataset_fn(input_context):
                batch_size = input_context.get_per_replica_batch_size(global_batch_size)
                ds = self._create_dateset(DatasetType.TRAIN, self.train_data_args, True, batch_size, True,
                                          2 * global_batch_size, num_workers, worker_idx)
                return ds

            def val_dataset_fn(input_context):
                batch_size = input_context.get_per_replica_batch_size(global_batch_size)
                ds = self._create_dateset(DatasetType.VALIDATION, self.val_data_args, True, batch_size, True,
                                          2 * global_batch_size, num_workers, worker_idx)
                return ds

            train_ds = dist_strategy.experimental_distribute_datasets_from_function(train_dataset_fn)

            if val_ds is not None:
                val_ds = dist_strategy.experimental_distribute_datasets_from_function(val_dataset_fn)
            steps_per_execution = self.train_args.get('steps_per_execution', 1)
            print("set steps_per_execution={} under tmeps training mode".format(steps_per_execution))
        else:
            steps_per_execution = self.train_args.get('steps_per_execution', 1)
            print("set steps_per_execution={}".format(steps_per_execution))

        epochs = self.train_args.get('epochs')
        if not epochs or epochs < 0:
            epochs = 1
            print("epochs not properly set, default to 1")

        model_call_exclude_input_index = self.train_args.get("model_call_exclude_input_index")
        input_squeeze = self.train_args.get("input_squeeze", False)
        with dist_strategy.scope():
            model = self._create_model()
            loss_scaled = is_using_mixed_precision(model)
            optimizer = self._create_optimizer(loss_scaled)
            losses, loss_weights = self._create_losses(model)
            metrics = self._create_metrics(model)
            callbacks = self._create_callbacks()
            if tf.__version__ < TF_REF_VERSION:
                model.compile(optimizer=optimizer, loss=losses, metrics=metrics, loss_weights=loss_weights)
            else:
                model.compile(optimizer=optimizer, loss=losses, metrics=metrics, loss_weights=loss_weights,
                              steps_per_execution=steps_per_execution)

            multi_optimizer = False
            if isinstance(optimizer, (list, tuple)):
                print("user specified multiple optimizers: {}".format(optimizer))
                multi_optimizer = True

            # 设置是否使用PCGrad
            use_pcgrad = False
            if isinstance(optimizer, PCGrad):
                num_tasks = 1 if not isinstance(losses, (list, tuple)) else len(losses)
                optimizer.set_num_tasks(num_tasks)
                print("use pcgrad optimizer: {}, num_tasks: {}".format(optimizer, optimizer.num_tasks))
                use_pcgrad = True

            # 这里根据训练流程的配置（是否有特殊metric，是否使用PCGrad等）来对模型的train_step等流程进行改写
            self._patch_model(model, metrics, losses, loss_weights, model_call_exclude_input_index, input_squeeze,
                              multi_optimizer, use_pcgrad)

        # 开始训练
        st = time.perf_counter()
        print("start training model under compile-fit mode...")
        model.fit(train_ds, epochs=epochs, verbose=2, callbacks=callbacks, validation_data=val_ds,
                  validation_freq=1, steps_per_epoch=steps_per_epoch,
                  validation_steps=validation_steps if val_ds is not None else None)
        print("model training finished, cost {}s".format(time.perf_counter() - st))

        return model

    def _train_custom_loop(self, dist_strategy: tf.distribute.Strategy, global_batch_size, steps_per_epoch,
                           validation_steps):
        """
        启动custom_loop训练流程

        Args:
            dist_strategy (tf.distribute.Strategy): 使用的分布式Strategy
            global_batch_size (int): 全局batch大小
            steps_per_epoch (int): 每个epoch的step数
            validation_steps (int): 每个epoch的验证step数

        Returns:
            tf.keras.Model: 训练好的模型实例
        """

        is_distributed = is_distributed_strategy(dist_strategy)

        is_ps_dist = isinstance(dist_strategy, (tf.distribute.experimental.ParameterServerStrategy,
                                                tf.compat.v1.distribute.experimental.ParameterServerStrategy))

        train_ds = self._create_dateset(DatasetType.TRAIN, self.train_data_args, is_distributed,
                                        global_batch_size, is_distributed)
        if train_ds is None:
            print("failed to load training data, exit, train_data_args={}".format(self.train_data_args))
            return None
        val_ds = self._create_dateset(DatasetType.VALIDATION, self.val_data_args, is_distributed,
                                      global_batch_size, is_distributed)

        if is_ps_dist:
            def train_dataset_fn(input_context):
                batch_size = input_context.get_per_replica_batch_size(global_batch_size)
                ds = self._create_dateset(DatasetType.TRAIN, self.train_data_args, True, batch_size, True,
                                          2 * global_batch_size, input_context.num_input_pipelines,
                                          input_context.input_pipeline_id)
                return ds

            def val_dataset_fn(input_context):
                batch_size = input_context.get_per_replica_batch_size(global_batch_size)
                ds = self._create_dateset(DatasetType.VALIDATION, self.val_data_args, True, batch_size, True,
                                          2 * global_batch_size, input_context.num_input_pipelines,
                                          input_context.input_pipeline_id)
                return ds

            @tf.function
            def per_worker_train_dataset_fn():
                return dist_strategy.distribute_datasets_from_function(train_dataset_fn)

            @tf.function
            def per_worker_val_dataset_fn():
                return dist_strategy.distribute_datasets_from_function(val_dataset_fn)

            # 在TF PS方式下，需要有一个coordinator（由chief角色执行）来向worker分发训练step
            coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(dist_strategy)
            train_ds = coordinator.create_per_worker_dataset(per_worker_train_dataset_fn)
            if val_ds is not None:
                val_ds = coordinator.create_per_worker_dataset(per_worker_val_dataset_fn)

            steps_per_execution = self.train_args.get('steps_per_execution', 10)
            print("set steps_per_execution={} under ps training mode".format(steps_per_execution))
        else:
            train_ds = dist_strategy.experimental_distribute_dataset(train_ds)
            if val_ds is not None:
                val_ds = dist_strategy.experimental_distribute_dataset(val_ds)

        epochs = self.train_args.get('epochs')
        if not epochs or epochs < 0:
            epochs = 1
            print("epochs not properly set, default to 1")

        model_call_exclude_input_index = self.train_args.get("model_call_exclude_input_index")
        input_squeeze = self.train_args.get("input_squeeze", False)

        def __training_loop():
            """
            训练循环
            """
            with dist_strategy.scope():
                model = self._create_model()
                callbacks = self._create_callbacks()
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    model=model,
                    epochs=epochs)

                input_modifier = InputsModifier(model_call_exclude_input_index, input_squeeze)

                loss_objects, loss_weights = self._create_losses(model, reduction=tf.keras.losses.Reduction.SUM)

                loss_scaled = is_using_mixed_precision(model)
                optimizer = self._create_optimizer(loss_scaled)
                train_metrics = self._create_metrics(model, name_prefix='train')
                test_metrics = self._create_metrics(model, name_prefix='val') if val_ds else []

                multi_optimizer = False
                if isinstance(optimizer, (list, tuple)):
                    print("user specified multiple optimizers: {}".format(optimizer))
                    multi_optimizer = True

            def compute_loss(labels, predictions, sample_weights):
                if isinstance(loss_objects, (list, tuple)):
                    loss_vals = []
                    for loss_object, loss_weight in zip(loss_objects, loss_weights):
                        loss = loss_object(labels, predictions, sample_weights)
                        loss_vals.append(loss*loss_weight)
                    per_example_loss = tf.add_n(loss_vals, name='train_custom_loop/add_all_losses')
                    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
                per_example_loss = loss_objects(labels, predictions, sample_weights)
                return per_example_loss / global_batch_size

            def update_component_metrics(y_true_comp, y_pred_comp, metric_comp, sw):
                if isinstance(metric_comp, (tuple, list)):
                    for m in metric_comp:
                        if isinstance(m, ExtendedMetric):
                            m.update_state(y_true_comp, y_pred_comp, sw, x)
                        else:
                            m.update_state(y_true_comp, y_pred_comp, sw)
                else:
                    if isinstance(metric_comp, ExtendedMetric):
                        metric_comp.update_state(y_true_comp, y_pred_comp, sw, x)
                    else:
                        metric_comp.update_state(y_true_comp, y_pred_comp, sw)

            def train_step(x, y, sample_weights):
                with tf.GradientTape() as tape:
                    x = input_modifier.modify_inputs(x)
                    predictions = model(x, training=True)
                    loss = compute_loss(y, predictions, sample_weights)
                if multi_optimizer:
                    trainable_var_groups, _ = call_user_module(self.user_module, AWFUserFunc.GROUP_TRAINABLE_VARS,
                                                               True, False, (list, tuple), model=model)
                    ExtendedTrainStep.multi_optimizer_minimize(dist_strategy, optimizer, tape, loss,
                                                               trainable_var_groups)
                else:
                    ExtendedTrainStep.single_optimizer_minimize(dist_strategy, optimizer, tape, loss,
                                                                model.trainable_variables)

                if train_metrics:
                    if isinstance(y, (tuple, list)):
                        for y_i, y_pred_i, metric_i in zip(y, predictions, train_metrics):
                            update_component_metrics(y_i, y_pred_i, metric_i, sample_weights)
                    elif isinstance(y, dict):
                        for output_name, y_i in y.items():
                            y_pred_i = predictions[output_name]
                            metric_i = train_metrics[output_name]
                            update_component_metrics(y_i, y_pred_i, metric_i, sample_weights)
                    else:
                        update_component_metrics(y, predictions, train_metrics, sample_weights)
                return loss

            def test_step(x, y, sample_weights):
                x = input_modifier.modify_inputs(x)
                predictions = model(x, training=False)
                if test_metrics:
                    if isinstance(y, (tuple, list)):
                        for y_i, y_pred_i, metric_i in zip(y, predictions, test_metrics):
                            update_component_metrics(y_i, y_pred_i, metric_i, sample_weights)
                    elif isinstance(y, dict):
                        for output_name, y_i in y.items():
                            y_pred_i = predictions[output_name]
                            metric_i = test_metrics[output_name]
                            update_component_metrics(y_i, y_pred_i, metric_i, sample_weights)
                    else:
                        update_component_metrics(y, predictions, test_metrics, sample_weights)

            if is_ps_dist:
                @tf.function
                def ps_distributed_train_step(ds_iterator):
                    for _ in range(steps_per_execution):
                        r = next(ds_iterator)
                        x, y, w = data_adapter.unpack_x_y_sample_weight(r)
                        losses = dist_strategy.run(train_step, args=(x, y, w))
                    return dist_strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

                @tf.function
                def ps_distributed_eval_step(ds_iterator):
                    r = next(ds_iterator)
                    x, y, w = data_adapter.unpack_x_y_sample_weight(r)
                    dist_strategy.run(test_step, args=(x, y, w))

                st = time.perf_counter()
                print("start training model under custom loop mode(PS)...")
                train_logs = None
                callbacks.on_train_begin()
                for epoch in range(epochs):
                    epoch_st = time.perf_counter()
                    tf.print("epoch {}/{} begin".format(epoch + 1, epochs))
                    callbacks.on_epoch_begin(epoch)
                    epoch_logs = {}
                    train_ds_iter = iter(train_ds)
                    for step_idx in range(steps_per_epoch):
                        coordinator.schedule(ps_distributed_train_step, args=(train_ds_iter, ))
                        # callbacks.on_train_batch_begin()
                        # logs = {m.name: m.result().numpy() for m in flatten_seq(train_metrics)} \
                        #     if train_metrics else {}
                        # callbacks.on_train_batch_end(step_idx*steps_per_execution, logs)
                    coordinator.join()
                    logs = {m.name: m.result().numpy() for m in flatten_seq(train_metrics)} \
                        if train_metrics else {}

                    for m in flatten_seq(train_metrics):
                        if isinstance(m, CumulationMetric):
                            logs[m.name] = m.final_result()
                    epoch_logs.update(logs)

                    if val_ds is not None:
                        val_ds_iter = iter(val_ds)
                        callbacks.on_test_begin()
                        for val_step_index in range(validation_steps):
                            # callbacks.on_test_batch_begin(test_batch_idx)
                            coordinator.schedule(ps_distributed_eval_step, args=(val_ds_iter, ))
                            # logs = {m.name: m.result().numpy() for m in flatten_seq(test_metrics)} \
                            #     if test_metrics else {}
                            # callbacks.on_test_batch_end(test_batch_idx, logs)
                        coordinator.join()
                        logs = {m.name: m.result().numpy() for m in flatten_seq(test_metrics)} \
                            if test_metrics else {}

                        for m in flatten_seq(test_metrics):
                            if isinstance(m, CumulationMetric):
                                logs[m.name] = m.final_result()
                        callbacks.on_test_end(logs)
                        epoch_logs.update(logs)

                    callbacks.on_epoch_end(epoch, epoch_logs)
                    train_logs = epoch_logs
                    tf.print("epoch {}/{} end, cost {}s, {}".format(epoch + 1, epochs,
                                                                    time.perf_counter()-epoch_st,
                                                                    epoch_logs))

                    for m in flatten_seq(train_metrics + test_metrics):
                        m.reset_states()
                    if model.stop_training:
                        break
                callbacks.on_train_end(train_logs)
                print("model training(PS) finished, cost {}s".format(time.perf_counter() - st))
                return model

            else:
                @tf.function
                def distributed_train_step(x, y, w):
                    per_replica_losses = dist_strategy.run(train_step, args=(x, y, w))
                    return dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

                @tf.function
                def distributed_test_step(x, y, w):
                    dist_strategy.run(test_step, args=(x, y, w))

                st = time.perf_counter()
                print("start training model under custom loop mode...")
                train_logs = None
                callbacks.on_train_begin()
                for epoch in range(epochs):
                    epoch_st = time.perf_counter()
                    tf.print("epoch {}/{} begin".format(epoch + 1, epochs))
                    callbacks.on_epoch_begin(epoch)
                    train_batch_idx = 0
                    epoch_logs = {}
                    for r in train_ds:
                        x, y, w = data_adapter.unpack_x_y_sample_weight(r)
                        callbacks.on_train_batch_begin(train_batch_idx)
                        distributed_train_step(x, y, w)
                        logs = {m.name: m.result().numpy() for m in flatten_seq(train_metrics)} \
                            if train_metrics else {}
                        callbacks.on_train_batch_end(train_batch_idx, logs)
                        train_batch_idx += 1
                        # if steps_per_epoch and 0 < steps_per_epoch <= train_batch_idx:
                        #     break

                    for m in flatten_seq(train_metrics):
                        if isinstance(m, CumulationMetric):
                            logs[m.name] = m.final_result()
                    epoch_logs.update(logs)

                    if val_ds:
                        callbacks.on_test_begin()
                        test_batch_idx = 0
                        for r in val_ds:
                            x, y, w = data_adapter.unpack_x_y_sample_weight(r)
                            callbacks.on_test_batch_begin(test_batch_idx)
                            distributed_test_step(x, y, w)
                            logs = {m.name: m.result().numpy() for m in flatten_seq(test_metrics)} \
                                if test_metrics else {}
                            callbacks.on_test_batch_end(test_batch_idx, logs)
                            test_batch_idx += 1
                            # if validation_steps and 0 < validation_steps <= test_batch_idx:
                            #     break

                        for m in flatten_seq(test_metrics):
                            if isinstance(m, CumulationMetric):
                                logs[m.name] = m.final_result()
                        callbacks.on_test_end(logs)
                        epoch_logs.update(logs)

                    callbacks.on_epoch_end(epoch, epoch_logs)
                    train_logs = epoch_logs
                    tf.print("epoch {}/{} end, cost {}s, {}".format(epoch + 1, epochs, time.perf_counter() - epoch_st,
                                                                    epoch_logs))

                    for m in flatten_seq(train_metrics + test_metrics):
                        m.reset_states()
                    if model.stop_training:
                        break
                callbacks.on_train_end(train_logs)
                print("model training finished, cost {}s".format(time.perf_counter() - st))
                return model

        if tf.__version__ < TF_REF_VERSION:
            from tensorflow.python.distribute import \
                distribute_coordinator_context as dc_context
            from tensorflow.python.distribute import multi_worker_util
            from tensorflow.python.distribute.distribute_coordinator import \
                _WorkerContext as WC

            task_type, task_index = self.get_task_info()
            cluster_info = self.get_cluster_info()
            print("cluster_info={}, task_type='{}', task_index={}".format(cluster_info, task_type, task_index))
            worker_context = WC(
                strategy=dist_strategy,
                cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_info) if cluster_info else None,
                task_type=task_type, task_id=task_index)

            with worker_context:
                print("current_worker_context={}".format(dc_context.get_current_worker_context()))
                return __training_loop()

        return __training_loop()

    def _estimator_model_fn(self, need_val, global_batch_size, features, labels, mode, params, config):
        model = self._create_model()
        predictions = model(features)

        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

        loss_objects, loss_weights = self._create_losses(model, reduction=tf.keras.losses.Reduction.SUM)
        if isinstance(loss_objects, (list, tuple)):
            loss_vals = []
            for loss_object, loss_weight in zip(loss_objects, loss_weights):
                loss = loss_object(labels, predictions)
                loss_vals.append(loss*loss_weight)
            per_example_loss = tf.add_n(loss_vals, name='train_estimator/add_all_losses')
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        else:
            loss = loss_objects(labels, predictions) / global_batch_size

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss_scaled = is_using_mixed_precision(model)
            optimizer = self._create_optimizer(loss_scaled, v1=True)
            metrics = None
            if need_val:
                metrics = self._create_metrics(model, is_training=True, v1=True)
                assert isinstance(metrics, dict)
                metrics = {n: m(labels, predictions) for n, m in metrics.items()}

            try:
                from tensorflow_recommenders_addons.dynamic_embedding import \
                    DynamicEmbeddingOptimizer
                optimizer = DynamicEmbeddingOptimizer(optimizer)
                print("wrapped optimizer {} with DynamicEmbeddingOptimizer".format(optimizer))
            except Exception as e:
                raise e
            train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.EVAL:
            metrics = self._create_metrics(model, is_training=False, v1=True)
            assert isinstance(metrics, dict)
            metrics = {n: m(labels, predictions) for n, m in metrics.items()}
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        raise NotImplementedError(f"unknown estimator mode '{mode}'")

    def _train_estimator(self, dist_strategy: tf.distribute.Strategy, global_batch_size, steps_per_epoch,
                         validation_steps):
        """
        启动estimator训练流程

        Args:
            dist_strategy (tf.distribute.Strategy): 使用的分布式Strategy
            global_batch_size (int): 全局batch大小
            steps_per_epoch (int): 每个epoch的step数
            validation_steps (int): 每个epoch的验证step数

        Returns:
            tf.estimator.Estimator: 训练好的estimator
        """
        is_distributed = is_distributed_strategy(dist_strategy)
        epochs = self.train_args.get('epochs')
        if not epochs or epochs < 0:
            epochs = 1
            print("epochs not properly set, default to 1")

        val_ds = self._create_dateset(DatasetType.VALIDATION, self.val_data_args, False, global_batch_size, False)
        need_val = val_ds is not None

        def __input_fn(ds_type):
            data_args = self.train_data_args if ds_type == DatasetType.TRAIN else self.val_data_args
            train_ds = self._create_dateset(ds_type, data_args, is_distributed, global_batch_size, is_distributed)
            if not is_distributed and ds_type == DatasetType.TRAIN:
                train_ds = train_ds.repeat(epochs)
            return train_ds

        def __model_fn(features, labels, mode, params, config):
            return self._estimator_model_fn(need_val, global_batch_size, features, labels, mode, params, config)

        if is_distributed:
            steps = epochs*steps_per_epoch
            print(f"set steps={steps}")
        else:
            steps = None

        ckpt_dir = make_abs_or_data_path("checkpoints", self.export_path, self.pack_path)
        run_config = tf.estimator.RunConfig(model_dir=ckpt_dir,
                                            train_distribute=dist_strategy if tf.distribute.has_strategy() else None,
                                            protocol='grpc')

        estimator = tf.estimator.Estimator(model_fn=__model_fn, config=run_config, params=self.model_args)

        st = time.perf_counter()
        if need_val:
            print("start training and evaluating model under estimator mode...")
            train_spec = tf.estimator.TrainSpec(lambda: __input_fn(DatasetType.TRAIN), max_steps=steps)
            eval_spec = tf.estimator.EvalSpec(lambda: __input_fn(DatasetType.VALIDATION), steps=validation_steps)
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        else:
            print("start training model under estimator mode...")
            estimator.train(lambda: __input_fn(DatasetType.TRAIN), max_steps=steps)
        print("model training finished, cost {}s".format(time.perf_counter() - st))
        return estimator

    def _setup_distribute_context(self, config_args, v1=False):
        has_gpu = tf.test.is_gpu_available()
        dist_type = self.get_distributed_type()
        tf.config.set_soft_device_placement(True)

        inter_op_paral = config_args.get('inter_op_paral', 0)
        intra_op_paral = config_args.get('intra_op_paral', 0)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_paral)
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_paral)
        if dist_type in [DistributionType.NONE, DistributionType.SINGLE_WORKER]:
            # 单机情况下，如果包含GPU，则使用MirroredStrategy以适应多卡情况，否则使用默认Strategy
            if has_gpu:
                dist_strategy = tf.distribute.MirroredStrategy()
            else:
                dist_strategy = tf.distribute.get_strategy()
        else:
            if dist_type == DistributionType.PS:
                # tf2.6之前的版本对PSStrategy支持的比较有限，因此这里对2.6版本之前的tf禁用了PS训练方式
                # 如果要使用PS，可以在配置中将tf_ver设置成tf2.6以切换到tf2.6版本的镜像
                if tf.__version__ < TF_REF_VERSION:
                    raise RuntimeError("tf version does not support PS distribution with keras,"
                                       " current tf version is '{}'".format(tf.__version__))
                cluster_info = self.tf_config.get('cluster', {})
                pss = cluster_info.get('ps', [])
                num_pss = len(pss)
                var_partition_policy = self.train_args.get('var_partition_policy', 'minsize:256k').lower().strip()
                if var_partition_policy.startswith('minsize'):
                    fields = var_partition_policy.split(":")
                    if len(fields) != 2:
                        min_shard_size = parse_size('256k')
                        print("default shard size to 256k for MinSizePartitioner")
                    else:
                        min_shard_size = parse_size(fields[1])
                        if min_shard_size is None or min_shard_size <= 0:
                            print("WARNING: invalid shard size '{}', auto set it to 256k for MinSizePartitioner"
                                  .format(var_partition_policy))
                        min_shard_size = parse_size('256k')
                    var_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
                        min_shard_size, num_pss)
                elif var_partition_policy.startswith('maxsize'):
                    fields = var_partition_policy.split(":")
                    if len(fields) != 2:
                        max_shard_size = parse_size('10M')
                        print("default shard size to 10M for MaxSizePartitioner")
                    else:
                        max_shard_size = parse_size(fields[1])
                        if max_shard_size is None or max_shard_size <= 0:
                            print("WARNING: invalid shard size '{}', auto set it to 10M for MaxSizePartitioner"
                                  .format(var_partition_policy))
                        max_shard_size = parse_size('10M')
                    var_partitioner = tf.distribute.experimental.partitioners.MaxSizePartitioner(
                        max_shard_size, num_pss)
                else:
                    var_partitioner = tf.distribute.experimental.partitioners.FixedShardsPartitioner(num_pss)

                print("var_partitioner={}".format(var_partitioner))

                self_task_type, self_task_id = self.get_task_info()
                if v1:
                    dist_strategy = tf.compat.v1.distribute.experimental.ParameterServerStrategy(
                        tf.compat.v1.distribute.cluster_resolver.SimpleClusterResolver(
                            tf.train.ClusterSpec(self.tf_config.get('cluster')),
                            task_type=self_task_type,
                            task_id=self_task_id,
                            rpc_layer='grpc')
                    )
                    print("created v1 ParameterServerStrategy for {}:{}".format(self_task_type, self_task_id))
                else:
                    dist_strategy = tf.distribute.experimental.ParameterServerStrategy(
                        tf.distribute.cluster_resolver.SimpleClusterResolver(
                            tf.train.ClusterSpec(self.tf_config.get('cluster')),
                            task_type=self_task_type,
                            task_id=self_task_id,
                            rpc_layer='grpc'),
                        variable_partitioner=var_partitioner)
            elif dist_type == DistributionType.TMEPS:
                from byteps.tensorflow.distribute import \
                    MirroredStrategy as BpsMirroredStrategy
                dist_strategy = BpsMirroredStrategy()
            else:
                mw_com = config_args.get('mw_com', '')
                mw_com = mw_com.strip().upper()
                if mw_com == "NCCL":
                    # 下面会根据是否包含GPU来自动设置集合通信库
                    if has_gpu:
                        optimizer_specs = config_args.get('optimizer')
                        multi_optimizer = isinstance(optimizer_specs, (list, tuple)) and len(optimizer_specs) > 1
                        if multi_optimizer:
                            print("WARING: there is bug in 'NCCL' backend when using multi-worker distribution"
                                  " to train multi-optimizer models, fallback to 'RING' to workaround")
                            mw_com = tf.distribute.experimental.CollectiveCommunication.RING
                        else:
                            mw_com = tf.distribute.experimental.CollectiveCommunication.NCCL
                    else:
                        print("WARNING: 'mw_com' set to 'NCCL' but found no GPU, fallback to 'RING'")
                        mw_com = tf.distribute.experimental.CollectiveCommunication.RING
                elif mw_com == "RING":
                    mw_com = tf.distribute.experimental.CollectiveCommunication.RING
                else:
                    mw_com = tf.distribute.experimental.CollectiveCommunication.AUTO
                print("multi-worker communication={}".format(mw_com))
                dist_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(mw_com)

        if config_args.get('verbose', False):
            tf.compat.v1.logging.set_verbosity(10)
        physical_devices = tf.config.list_physical_devices()
        logical_devices = tf.config.list_logical_devices()
        print("physical_devices:\n{}".format(physical_devices))
        print("logical_devices:\n{}".format(logical_devices))

        return dist_type, dist_strategy

    def run_train(self):
        """
        启动模型训练流程

        Raises:
            RuntimeError: _description_
            NotImplementedError: _description_
            RuntimeError: _description_

        Returns:
            list: [(模型路径, 模型名), ...]
        """
        print("{}: train_args={}".format(__file__, self.train_args))
        self._init_envs()

        tf_log_level = tf_log_level_from_string(self.train_args.get('tf_log_level', 'info'))
        if tf_log_level is not None:
            tf.compat.v1.logging.set_verbosity(tf_log_level)
            print("set tf log level to {}".format(tf_log_level))

        if self.train_args.get('log_device', False):
            tf.debugging.set_log_device_placement(True)
            print("turn on tf log device placement")

        dist_type = self.get_distributed_type()
        if dist_type == DistributionType.PS:
            task_type, task_id = self.get_task_info()
            if task_type in ['worker', 'ps']:
                print("will start server for '{}' {}".format(task_type, task_id))
                server = tf.distribute.Server(tf.train.ClusterSpec(self.tf_config.get('cluster')),
                                              job_name=task_type,
                                              task_index=task_id,
                                              protocol='grpc',
                                              start=True)
                server.join()
                return
        elif dist_type == DistributionType.TMEPS:
            print("TMEPS environments: ", flush=True)
            for k, v in os.environ.items():
                print("\t'{}'='{}'".format(k, v), flush=True)

            try:
                import byteps.tensorflow as bps
            except:
                raise RuntimeError("BPS is not supported under current os, please switch to\
                                   ubuntu by setting 'os_ver' to 'ubt'")
            task_type, task_id = self.get_task_info()
            if task_type in ['chief', 'ps']:
                print("will launch tmeps service for '{}' {}".format(task_type, task_id))
                subprocess.check_call("bpslaunch", env=os.environ.copy(), stdout=sys.stdout,
                                      stderr=sys.stderr, shell=True)
                return

            print("about to initialize bps", flush=True)
            bps.init()
            print("bps initialize finished", flush=True)

        if self.train_args.get("gpu_mem_growth", False):
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            tf.compat.v1.Session(config=config)

            gpus = tf.config.experimental.list_physical_devices('GPU')
            for i, gpu in enumerate(gpus):
                tf.config.experimental.set_memory_growth(gpu, True)
                print("enabled memory growth for gpu {} {}: {}".format(
                    i, gpu, tf.config.experimental.get_memory_growth(gpu)),
                    flush=True)

        train_type = self.train_args.get('train_type', RunType.COMPILE_FIT)
        if not train_type or not train_type.strip():
            train_type = RunType.COMPILE_FIT
            print("'train_type' not set, default to {}".format(train_type))

        train_type = train_type.strip().lower()
        if train_type not in [RunType.COMPILE_FIT, RunType.CUSTOM_LOOP, RunType.ESTIMATOR]:
            raise NotImplementedError("unknown train type '{}', supported are {}"
                                      .format(train_type, [RunType.COMPILE_FIT, RunType.CUSTOM_LOOP,
                                                           RunType.ESTIMATOR]))

        _, dist_strategy = self._setup_distribute_context(self.train_args, train_type == RunType.ESTIMATOR)

        mix_precision = self.train_args.get('mix_precision', False)
        if mix_precision:
            # 如果使用混合精度，这里把全局的精度策略设置为mixed_float16
            print("use mixed precision training")
            if tf.__version__ < TF_REF_VERSION:
                dtype_policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
                tf.keras.mixed_precision.experimental.set_policy(dtype_policy)
            else:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")

        batch_size = self.train_args.get('batch_size', 128)
        if batch_size <= 0:
            print("'batch_size' inproperly set to {}, changed to 128".format(batch_size))
            batch_size = 128

        # 根据当前参与计算的节点数及设备数，计算全局batch大小
        global_batch_size = batch_size * dist_strategy.num_replicas_in_sync
        print("train: dist_type={}, dist_strategy={}, dist_strategpy.extended={}, num_replicas_in_sync={},"
              " batch_size={}, global_batch_size={}".format(dist_type, dist_strategy, dist_strategy.extended,
                                                            dist_strategy.num_replicas_in_sync,
                                                            batch_size, global_batch_size))

        if dist_type in [DistributionType.MULTI_WORKER, DistributionType.PS, DistributionType.TMEPS]:
            # 对于多机的分布式训练，因为需要保证每个机器的batch大小均匀，
            # 需要设置训练样本数num_samples及验证样本数num_val_samples（如果有验证集的话），
            # 以便于计算每个epoch的训练step数及验证step数。因为dataset会做repeat，
            # 所以num_samples和num_val_samples不必要非常精确，量级上大致差不多就行

            num_samples = self.train_args.get('num_samples')
            if not num_samples or num_samples < 0:
                raise RuntimeError("'num_samples' must be set when using distribution training")

            if num_samples < global_batch_size:
                print("WARNING: number of samples {} < num_workers*batch_size {}, auto set steps_per_epoch=1"
                      .format(num_samples, global_batch_size))
                steps_per_epoch = 1
            else:
                steps_per_epoch = num_samples // global_batch_size
                if dist_type == DistributionType.TMEPS:
                    steps_per_epoch = steps_per_epoch // self.get_num_workers()
                print("set 'steps_per_epoch'={}".format(steps_per_epoch))

            num_val_samples = self.train_args.get('num_val_samples', 0)
            if num_val_samples < global_batch_size:
                print("WARNING: number of validation samples {} < num_workers*batch_size {},"
                      " auto set validation_steps=1".format(num_val_samples, global_batch_size))
                validation_steps = 1
            else:
                validation_steps = num_val_samples // global_batch_size
                print("set 'validation_steps'={}".format(validation_steps))
        else:
            # 单机情况下，每个epoch会遍历完数据集，此时不需要设置num_samples及num_val_samples
            print("non-distributed training, ignore 'num_samples' and 'validation_steps'")
            steps_per_epoch = None
            validation_steps = None

        if train_type == RunType.COMPILE_FIT:
            trained_model = self._train_compile_fit(dist_strategy, global_batch_size, steps_per_epoch, validation_steps)
        elif train_type == RunType.CUSTOM_LOOP:
            trained_model = self._train_custom_loop(dist_strategy, global_batch_size, steps_per_epoch, validation_steps)
        elif train_type == RunType.ESTIMATOR:
            # 对于estimator方式，需要禁用掉动态图模式，否则会报错
            tf.compat.v1.disable_eager_execution()
            trained_model = self._train_estimator(dist_strategy, global_batch_size, steps_per_epoch, validation_steps)

        if trained_model is not None:
            # 训练完成，准备保存模型
            save_path = self.train_args.get("save_path")
            if not save_path or not save_path.strip():
                save_path = "saved_model"
            else:
                save_path = save_path.strip()
            save_path = make_abs_or_data_path(save_path, self.export_path, self.pack_path)
            if not self.is_chief():
                task_type, task_index = self.get_task_info()
                save_path = os.path.join(save_path, ".{}-{}_tmp".format(task_type, task_index))
            if not os.path.isdir(save_path):
                os.makedirs(save_path, exist_ok=True)
                print("created model saving dir '{}', is_chief={}".format(save_path, self.is_chief()))

            # estimator和keras model的保存方式是不一样的
            # 目前estimator暂时只支持一次保存单个模型，keras model则支持一次保存多个模型
            saved_models = []
            if train_type != RunType.ESTIMATOR:
                # keras model支持保存多个模型
                model_name = trained_model.name or self.model_args.get('name', '').strip() \
                    or split_file_name(self.user_py_file)[1]
                with dist_strategy.scope():
                    print("mode to save under distribution stratege={}".format(dist_strategy))
                    model_save_infos = self._model_to_save(trained_model)
                for i, (model_to_save, signature, save_options) in enumerate(model_save_infos):
                    model_path = save_path
                    if len(model_save_infos) > 1:
                        model_name = model_to_save.name or (model_name + "-{}".format(i))
                        model_path = os.path.join(save_path, model_name)
                    model_to_save.save(model_path, signatures=signature, options=save_options, include_optimizer=False)
                    print("saved model '{}' to '{}', is_chief={}".format(model_name, model_path, self.is_chief()))
                    saved_models.append((model_path, model_name))
            else:
                # estimator只支持一次保存一个模型
                model_name = trained_model.params.get('name') or split_file_name(self.user_py_file)[1]
                input_ph_dict, _ = call_user_module(self.user_module, AWFUserFunc.INPUT_PLACEHOLDERS, True, False, dict,
                                                    model_input_config_file=None, model=self._create_model())
                input_recevier_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(input_ph_dict)
                model_path = trained_model.export_saved_model(save_path, input_recevier_fn)
                model_path = model_path.decode('utf-8')
                print("estimator saved model '{}' to '{}'".format(model_name, model_path))
                saved_models.append((model_path, model_name))

            return saved_models
        return None

    def _load_model(self, model_path, model_name, purpose=None) -> tf.keras.Model:
        """
        调用用户回调awf_load_model_fn，加载从指定路径加载模型，用户回调需要返回加载后的
        tf.keras.Model对象

        Args:
            model_path (str): 要加载的模型路径，此参数会传递给用户回调
            model_name (str): 模型名称，此参数会传递给用户回调
            purpose (str, optional): 加载模型的目的，evaluate表示用于评估，predict表示用于预测. Defaults to None.

        Returns:
            tf.keras.Model: 加载的模型实例对象
        """

        args = self.load_model_args or {}
        args.update({'path': model_path, 'name': model_name})
        inject_args = {'pack_path': self.pack_path, 'data_path': self.export_path, 'export_path': self.export_path,
                       "purpose": purpose}
        model, ia = call_user_module(self.user_module, AWFUserFunc.LOAD_MODEL, False, True, tf.keras.Model,
                                     inject_args=inject_args, **args)
        if model is None and ia is None:
            print("user function '{}' did not return a model, will try to load it from '{}'"
                  .format(AWFUserFunc.LOAD_MODEL, model_path))
            model = tf.keras.models.load_model(model_path, compile=False)
            print("loaded model '{}' from '{}': {}".format(model_name, model_path, model))
        elif model is not None:
            print("loaded model '{}' from '{}' by user function '{}': {}"
                  .format(model_name, model_path, AWFUserFunc.LOAD_MODEL, model))
        return model

    def _eval_keras_model(self, model_specs, dist_strategy: tf.distribute.Strategy, global_batch_size):
        """
        keras模型评估

        Args:
            model_specs (list): 要评估的模型列表，[{"path": "模型路径", "name": "模型名"}, ...]
            dist_strategy (tf.distribute.Strategy): 评估时使用的分布式strategy
            global_batch_size (int): 评估的全局batch大小

        Raises:
            RuntimeError: _description_

        Returns:
            list: [(模型名，模型路径，评估结果),...]，评估结果的的格式为{"指标名": 指标值, ...}
        """

        is_distributed = is_distributed_strategy(dist_strategy)

        test_ds = self._create_dateset(DatasetType.TEST, self.test_data_args, is_distributed,
                                       global_batch_size, is_distributed,
                                       check_return_type=(tf.data.Dataset, list, tuple, dict))
        if not test_ds:
            print("got no test data, exit")
            return None

        if is_distributed:
            # 与训练时类似，多机评估时，因为需要保证每个机器的batch大小均匀，
            # 也需要设定评估的样本数num_test_samples, 以便于计算step数。
            # num_test_samples不必要非常精确，量级上大致差不多就行

            num_test_samples = self.evaluate_args.get('num_test_samples')
            if not num_test_samples or num_test_samples < 0:
                raise RuntimeError("'num_test_samples' must be set when using distribution evaluation")

            if num_test_samples < global_batch_size:
                print("WARNING: num_test_samples {} < num_workers*batch_size {}, auto set steps_per_epoch=1"
                      .format(num_test_samples, global_batch_size))
                eval_steps = 1
            else:
                eval_steps = num_test_samples // global_batch_size
                print("set 'eval_steps'={}".format(eval_steps))
        else:
            print("non-distributed evaluating, ignore 'num_test_samples'")
            eval_steps = None

        model_call_exclude_input_index = self.evaluate_args.get("model_call_exclude_input_index")
        input_squeeze = self.evaluate_args.get("input_squeeze", False)

        metric_specs = self.evaluate_args.get('metrics')
        if not metric_specs:
            print("no model metrics specified, exit")
            return None

        with dist_strategy.scope():
            loaded_models = []
            dedup = set()
            for i, spec in enumerate(model_specs):
                model_path = spec.get('path')
                if not model_path or not model_path.strip():
                    print("{}th evaluation 'models.path' not set, ignore it".format(i))
                    continue
                model_path = make_abs_or_data_path(model_path.strip(), self.export_path, self.pack_path)
                if model_path in dedup:
                    print("{}th evaluation model path '{}' duplicated, ignore it".format(i, model_path))
                    continue
                dedup.add(model_path)
                set_model_name = spec.get('name', '').strip()
                if not set_model_name:
                    default_name = 'model_%s' % i
                load_name = set_model_name or default_name
                model = self._load_model(model_path, load_name, purpose="evaluate")
                if model is None:
                    print("no model '{}' to load from '{}'".format(load_name, model_path))
                    continue

                optimizer = self._create_optimizer()
                losses, _ = self._create_losses(model, is_training=False)
                metrics = self._create_metrics(model, is_training=False)

                model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
                self._patch_model(model, metrics, losses, _, model_call_exclude_input_index, input_squeeze)
                loaded_models.append((set_model_name or model.name or default_name, model_path, model))

        if not loaded_models:
            print("no model to be evaluated, exit")
            return None

        def __reconstruct_evaluation_result(ret, name_prefix=None):
            rec_ret = {}
            for k, v in ret.items():
                rec_name = k if name_prefix is None else "{}({})".format(k, str(name_prefix))
                rec_ret[rec_name] = v
            return rec_ret

        eval_results = []
        for i, (name, path, model) in enumerate(loaded_models):
            st = time.perf_counter()
            if isinstance(test_ds, tf.data.Dataset):
                try:
                    eval_result = model.evaluate(test_ds, return_dict=True, steps=eval_steps)
                    eval_results.append((name, path, __reconstruct_evaluation_result(eval_result)))
                except Exception as e:
                    print("failed to evaluate model '{}' from path '{}' on val-dataset: {}\n{}"
                          .format(name, path, e, traceback.format_exc()))
            elif isinstance(test_ds, (list, tuple)):
                eval_result = {}
                for idx, ds in enumerate(test_ds):
                    try:
                        eval_result_per_ds = model.evaluate(ds, return_dict=True, steps=eval_steps)
                        eval_result.update(__reconstruct_evaluation_result(eval_result_per_ds, idx))
                    except Exception as e:
                        print("failed to evaluate model '{}' from path '{}' on {}th val-dataset: {}\n{}"
                              .format(name, path, idx, e, traceback.format_exc()))
                eval_results.append((name, path, eval_result))
            else:
                eval_result = {}
                for n, ds in test_ds.items():
                    try:
                        eval_result_per_ds = model.evaluate(ds, return_dict=True, steps=eval_steps)
                        eval_result.update(__reconstruct_evaluation_result(eval_result_per_ds, n))
                    except Exception as e:
                        print("failed to evaluate model '{}' from path '{}' on '{}' val-dataset: {}\n{}"
                              .format(name, path, n, e, traceback.format_exc()))
                eval_results.append((name, path, eval_result))
            print("evaluated {}th model '{}' from path '{}', cost {}s: {}"
                  .format(i, name, path, time.perf_counter() - st, eval_result))

        for name, path, model in loaded_models:
            if model is not None:
                del model
                print("released loaded model '{}' from '{}'".format(name, path))
        loaded_models.clear()
        del loaded_models
        return eval_results

    def _eval_estimator_model(self, model_specs, dist_strategy: tf.distribute.Strategy, global_batch_size):
        """
        estimator模型评估

        Args:
            model_specs (list): 要评估的模型列表，[{"path": "模型路径", "name": "模型名"}, ...]
            dist_strategy (tf.distribute.Strategy): 评估时使用的分布式strategy
            global_batch_size (int): 评估的全局batch大小

        Raises:
            RuntimeError: _description_

        Returns:
            list: [(模型名，模型路径，评估结果),...]，评估结果的的格式为{"指标名": 指标值, ...}
        """

        if not model_specs:
            print("no model to be evaluated, exit")
            return None
        assert len(model_specs) == 1, "support only 1 model evaluation under estimator mode, got %s" % model_specs

        is_distributed = is_distributed_strategy(dist_strategy)
        test_ds = self._create_dateset(DatasetType.TEST, self.test_data_args, False, global_batch_size, False)
        if not test_ds:
            print("got no test data, exit")
            return None

        if is_distributed:
            num_test_samples = self.evaluate_args.get('num_test_samples')
            if not num_test_samples or num_test_samples < 0:
                raise RuntimeError("'num_test_samples' must be set when using distribution evaluation")

            if num_test_samples < global_batch_size:
                print("WARNING: num_test_samples {} < num_workers*batch_size {}, auto set steps_per_epoch=1"
                      .format(num_test_samples, global_batch_size))
                eval_steps = 1
            else:
                eval_steps = num_test_samples // global_batch_size
                print("set 'eval_steps'={}".format(eval_steps))
        else:
            print("non-distributed evaluating, ignore 'num_test_samples'")
            eval_steps = None

        model_name = model_specs[0].get('name', '').strip() or 'model'
        model_path = model_specs[0].get('path', '').strip()
        if not model_path:
            print("WARNING: no model path is provided, skip model evaluation")
            return None
        model_path = make_abs_or_data_path(model_path, self.export_path, self.pack_path)
        if not os.path.isdir(model_path):
            print(f"WARNING: model path '{model_path}' is exists")
            return None

        def __input_fn():
            return self._create_dateset(DatasetType.TEST, self.test_data_args, is_distributed,
                                        global_batch_size, is_distributed)

        def __model_fn(features, labels, mode, params, config):
            return self._estimator_model_fn(False, global_batch_size, features, labels, mode, params, config)

        ckpt_dir = make_abs_or_data_path("checkpoints", self.export_path, self.pack_path)

        run_config = tf.estimator.RunConfig(model_dir=ckpt_dir,
                                            eval_distribute=dist_strategy if tf.distribute.has_strategy() else None)

        estimator = tf.estimator.Estimator(model_fn=__model_fn, config=run_config, params=self.evaluate_args)

        print(f"start evaluating model '{model_name}' in '{model_path}' under estimator mode,\
            steps='{eval_steps}', checkpoint path='{ckpt_dir}'...")
        st = time.perf_counter()
        eval_ret = estimator.evaluate(__input_fn, steps=eval_steps)
        print("model evaluation finished, cost {}s".format(time.perf_counter() - st))
        recons_eval_ret = {}
        for k, v in eval_ret.items():
            recons_eval_ret[k] = v.item()
        return [(model_name, model_path, recons_eval_ret)]

    def run_evaluate(self):
        """
        启动评估流程

        Raises:
            RuntimeError: _description_

        Returns:
            list: [(模型名，模型路径，评估结果),...]，评估结果的的格式为{"指标名": 指标值, ...}
        """

        print("{}: evaluate_args={}".format(__file__, self.evaluate_args))
        if not self.evaluate_args:
            print("evaluate_args not set, exit")
            return None

        model_specs = self.evaluate_args.get("models")
        if not model_specs:
            print("evaluation 'models' not set, exit")
            return None
        if not isinstance(model_specs, (list, dict)):
            raise RuntimeError("evaluation 'models' should be dict or list, got {}".format(model_specs))
        if isinstance(model_specs, dict):
            model_specs = [model_specs]

        dist_type, dist_strategy = self._setup_distribute_context(self.evaluate_args)

        batch_size = self.evaluate_args.get('batch_size', 128)
        if batch_size <= 0:
            print("evaluation 'batch_size' inproperly set to {}, changed to 128".format(batch_size))
            batch_size = 128
        global_batch_size = batch_size * dist_strategy.num_replicas_in_sync
        print("evaluation: dist_type={}, dist_strategy={}, dist_strategpy.extended={}, num_replicas_in_sync={},"
              " batch_size={}, global_batch_size={}".format(dist_type, dist_strategy, dist_strategy.extended,
                                                            dist_strategy.num_replicas_in_sync,
                                                            batch_size, global_batch_size))

        train_type = self.train_args.get('train_type', '').strip() or self.evaluate_args.get('train_type', '').strip()
        if train_type == RunType.ESTIMATOR:
            return self._eval_estimator_model(model_specs, dist_strategy, global_batch_size)
        return self._eval_keras_model(model_specs, dist_strategy, global_batch_size)

    def run_predict(self):
        """
        运行模型离线预测流程

        Returns:
            None: None
        """

        print("{}: predict_args={}".format(__file__, self.predict_args))

        if not self.predict_args:
            print("predict_args not set, exit")
            return None
        model_path = self.predict_args.get("model_path", '').strip()
        model_name = self.predict_args.get("model_name", '').strip()

        if not model_path or not os.path.isdir(model_path):
            print("model path '{}' is not a valid directory, exit".format(model_path))
            return None

        batch_size = self.predict_args.get('batch_size', 1024)
        if batch_size <= 0:
            print("predict 'batch_size' inproperly set to {}, changed to 1024".format(batch_size))
            batch_size = 1024
        pred_ds = self._create_dateset(DatasetType.PREDICTION, self.predict_data_args, False,
                                       batch_size, False, check_return_type=(tf.data.Dataset, list, tuple, dict))
        if not pred_ds:
            print("got no prediction data, exit")
            return None

        model = self._load_model(model_path, model_name, purpose="predict")
        if model is None:
            print("failed to load model '{}' from '{}'".format(model_name, model_path))
            return None

        result_file = self.predict_args.get('result_file', '').strip()
        result_field_delim = self.predict_args.get('result_field_delim', ' ')
        output_delim = self.predict_args.get('output_delim', ',')
        row_id_col = self.predict_args.get('row_id_col', '').strip()
        input_exclude_row_id = self.predict_args.get('input_exclude_row_id', True)
        row_format = self.predict_args.get('row_format')
        write_headers = self.predict_args.get('write_headers', True)
        if not result_file:
            result_file = 'predict.csv'
        result_file = make_abs_or_data_path(result_file, self.export_path, self.pack_path)

        st = time.perf_counter()
        print("start predicting use model from '{}', result file='{}'".format(model_path, result_file))
        write_counter = 0
        with open(result_file, 'w') as pf:
            head_wrote = False
            for batch in pred_ds:
                if row_id_col:
                    if input_exclude_row_id:
                        row_ids = batch.pop(row_id_col)
                    else:
                        row_ids = batch[row_id_col]
                predicts = model.predict(batch)
                if isinstance(predicts, tuple):
                    headers = list(map(lambda i: 'output_' + str(i), range(len(predicts))))
                elif isinstance(predicts, dict):
                    headers = list(predicts.keys())
                else:
                    headers = ['output']
                if row_id_col:
                    headers = [row_id_col] + headers

                if not head_wrote and write_headers:
                    # 结果文件写入列头
                    if row_format:
                        # 格式化列头，列头格式设置方式可以参考
                        # http://tapd.woa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc14
                        # 中关于row_format的说明
                        import re
                        headers_str = re.sub(r'\{([^{}]+)\}', lambda x: x.group(1), row_format)
                        pf.write(headers_str + '\n')
                    else:
                        pf.write(result_field_delim.join(headers) + '\n')
                    head_wrote = True

                if row_id_col:
                    # 拼接row_id和预测各field值
                    if isinstance(predicts, tuple):
                        ret_batch = zip(row_ids, *predicts)
                    elif isinstance(predicts, dict):
                        ret_batch = zip(row_ids, *tuple(predicts.values()))
                    else:
                        ret_batch = zip(row_ids, predicts)
                else:
                    if isinstance(predicts, tuple):
                        ret_batch = zip(*predicts)
                    elif isinstance(predicts, dict):
                        ret_batch = zip(*tuple(predicts.values()))
                    else:
                        ret_batch = zip(predicts)

                if not row_format:
                    row_format = ''
                    output_cnt = len(predicts) if isinstance(predicts, (tuple, dict)) else 1
                    if row_id_col:
                        row_format = '{row_id}'
                    for i in range(output_cnt):
                        if row_format:
                            row_format += result_field_delim
                        row_format += '{output_%s}' % i

                for row in ret_batch:
                    # 预测结果batch逐行写入结果文件
                    format_values = {}
                    if row_id_col:
                        if row[0].dtype == tf.string:
                            row_id = row[0].numpy().decode('utf8')
                        else:
                            row_id = str(row[0].numpy())
                        format_values['row_id'] = row_id
                        format_values[row_id_col] = row_id
                        row = row[1:]

                    for i, o in enumerate(row):
                        o_str = output_delim.join(map(str, o.reshape(-1)))
                        format_values['output_%s' % i] = o_str
                        header_name = headers[i + 1] if row_id_col else headers[i]
                        format_values[header_name] = o_str
                        if i == 0:
                            format_values['output'] = o_str

                    row_str = row_format.format(**format_values)
                    pf.write(row_str + '\n')
                    write_counter += 1
                    if write_counter > 0 and write_counter % 50000 == 0:
                        print("wrote {} prediction results, cost {}s".format(write_counter, time.perf_counter() - st))
        print("prediction finished, totally wrote {} results, cost {}s".format(write_counter, time.perf_counter() - st))
        del model
