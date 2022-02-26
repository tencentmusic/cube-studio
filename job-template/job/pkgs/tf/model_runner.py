
import json
import os
import sys
import time
import types
from typing import Union
import traceback
import copy
from .helperfuncs import TF_REF_VERSION

import tensorflow as tf
from tensorflow.python.eager import backprop, context
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import data_adapter

if tf.__version__ < TF_REF_VERSION:
    from tensorflow.python.keras.engine.training import enable_multi_worker, _keras_api_gauge, \
        _disallow_inside_tf_function
else:
    from tensorflow.python.keras.engine.base_layer import keras_api_gauge as _keras_api_gauge
    from tensorflow.python.keras.engine.training import _disallow_inside_tf_function


    def enable_multi_worker(worker_fn):
        return worker_fn
from tensorflow.python.keras.utils import version_utils, tf_utils
from tensorflow.python.profiler import trace

if tf.__version__ < TF_REF_VERSION:
    from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as lso
else:
    from tensorflow.python.keras.mixed_precision import loss_scale_optimizer as lso
from tensorflow.python.distribute import parameter_server_strategy

from .extend_metrics import ExtendedMetric, CumulationMetric, ExtendedMetricsContainer
from .extend_utils import PCGrad, ExtendedLossesContainer
from .helperfuncs import create_optimizer, create_loss, create_metric, create_grad_process
from ..constants import AWFUserFunc, DatasetType, DistributionType, KF_UI_META_FILE
from ..generators import flatten_seq
from ..utils import make_abs_or_data_path, make_abs_or_pack_path, expand_param, call_user_module, \
    split_file_name, parse_size

TRAIN_TYPE_COMPILE_FIT = "compile_fit"
TRAIN_TYPE_CUSTOM_LOOP = "custom_loop"


class InputsModifier(object):
    def __init__(self, model_call_exclude_input_index=None, squeeze=False):
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
        super(ExtendedTrainStep, self).__init__(model_call_exclude_input_index, squeeze)
        self.multi_optimizer = multi_optimizer
        self.user_module = user_module
        self.use_pcgrad = use_pcgrad

    def __call__(self, model_self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

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
                    with trace.Trace('TraceContext', graph_type='test', step_num=step):
                        callbacks.on_test_batch_begin(step)
                        tmp_logs = test_function(iterator)
                        if data_handler.should_sync:
                            context.async_wait()
                        logs = tmp_logs  # No error, now safe to assign to logs.
                        end_step = step + data_handler.step_increment
                        callbacks.on_test_batch_end(end_step, logs)

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

    def is_chief(self):
        if not self.tf_config:
            return True
        role, index = self._get_task_info()
        if role:
            role = role.strip().lower()
        if role in ['chief', 'master']:
            return True
        cluster = self._get_cluster_info()
        if 'chief' in cluster or 'master' in cluster:
            return False
        return role == 'worker' and index == 0

    def _get_task_info(self):
        if not self.tf_config:
            return None, None
        task_info = self.tf_config.get('task', {})
        if not task_info:
            return None, None
        return task_info.get('type'), task_info.get('index')

    def _get_cluster_info(self):
        if not self.tf_config:
            return None
        return self.tf_config.get('cluster')

    def _create_model(self) -> tf.keras.Model:
        continue_training = self.model_args.get("continue_training", False)
        continue_training_model_path = self.model_args.get("continue_training_model_path", "")
        if continue_training and not os.path.exists(continue_training_model_path):
            continue_training = False  # 首次增量训练从头开始训练
            print("continue_training_model_path '{}' is not exists, so change to normal training process"
                  .format(continue_training_model_path))

        inject_args = {'pack_path': self.pack_path, 'data_path': self.export_path, 'export_path': self.export_path}
        if continue_training:
            continue_training_model_name = self.model_args.get("name", "default")
            load_model_args = {'path': continue_training_model_path, 'name': continue_training_model_name}
            model, _ = call_user_module(self.user_module, AWFUserFunc.LOAD_MODEL, False, True, tf.keras.Model,
                                        inject_args=inject_args, **load_model_args)
        else:
            model, _ = call_user_module(self.user_module, AWFUserFunc.CRETAE_MODEL, True, False, tf.keras.Model,
                                        inject_args=inject_args, **self.model_args)
        return model

    def _create_dateset(self, dataset_type, dataset_args, repeat, global_batch_size, drop_remainder,
                        shuffle_buffer_size=None, num_shards=None, shard_index=None,
                        check_return_type=tf.data.Dataset) -> Union[tf.data.Dataset, list, dict, tuple]:

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
            if shuffle_buffer_size is not None and shuffle_buffer_size > 0:
                ds = ds.shuffle(shuffle_buffer_size)
                print("shuffled {} dataset with buffer_size {}".format(dataset_type, shuffle_buffer_size))

            if 'repeat' in injected_args:
                print("injected 'repeat' in user function '{}', will not do repeat".format(func_name))
            elif repeat:
                print("user function '{}' has no 'repeat' arg, repeated dataset".format(func_name, global_batch_size))
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

    def _create_optimizer(self, loss_scaled=False, **kwargs):
        def __create_one_optimizer(optimizer_detail):
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
                        print("type of optimizer not set, type default to '{}' and learning rate default to"
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
            optimizer = create_optimizer(optimizer_type, lr, **args)
            if optimizer is None:
                raise NotImplementedError("unsupported optimizer type '{}'".format(optimizer_type))

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
            if isinstance(detail, str):
                loss_type = detail.strip()
                if not loss_type:
                    raise RuntimeError("loss can not be empty string, is_training={}".format(is_training))
                args = kwargs or {}
                loss = create_loss(loss_type, **args)
                if loss is None:
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

    def _create_metrics(self, model, name_prefix=None, is_training=True, **kwargs):
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
                metric = create_metric(metric_type, name_prefix, **args)
                if metric is None:
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
                metric = create_metric(metric_type, name_prefix, **args)
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
            return metric

        metric_list = []
        metric_dict = {}
        if isinstance(metrics_detail, str) or (isinstance(metrics_detail, dict) and 'type' in metrics_detail):
            metric = __parse_single_metric(metrics_detail)
            metric_list.append(metric)
        elif isinstance(metrics_detail, dict):
            for output_name, metric_item in metrics_detail.items():
                if not isinstance(metric_item, (str, dict, list)):
                    raise RuntimeError("metric of output '{}' must be str or dict or list, got {}, is_training={}"
                                       .format(output_name, metric_item, is_training))
                if isinstance(metric_item, (str, dict)):
                    metric = __parse_single_metric(metric_item)
                    metric_dict[output_name] = metric
                else:
                    sub_metric_list = []
                    for j, sub_item in enumerate(metric_item):
                        if not isinstance(sub_item, (str, dict)):
                            raise RuntimeError("({}th metric of output '{}' must be str or dict, got {}, is_training={}"
                                               .format(j, output_name, sub_item, is_training))
                        sub_metric = __parse_single_metric(sub_item)
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
                    metric = __parse_single_metric(metric_item, index)
                    metric_list.append(metric)
                else:
                    sub_metric_list = []
                    for j, sub_item in enumerate(metric_item):
                        if not isinstance(sub_item, (str, dict)):
                            raise RuntimeError("({}, {})th metric must be str or dict, got {}, is_training={}"
                                               .format(i, j, sub_item, is_training))
                        sub_metric = __parse_single_metric(sub_item, index)
                        sub_metric_list.append(sub_metric)
                    metric_list.append(sub_metric_list)

        print("created metrics={}, name_prefix='{}', is_training={}".format(metric_list or metric_dict, name_prefix,
                                                                            is_training))
        return metric_list or metric_dict

    def _create_callbacks(self):
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

    def get_distributed_type(self):
        if not self.tf_config:
            return DistributionType.NONE
        cluster_info = self.tf_config.get('cluster', {})
        workers = cluster_info.get('worker', [])
        pss = cluster_info.get('ps', [])
        chiefs = cluster_info.get('chief', [])
        masters = cluster_info.get('master', [])
        if len(pss) > 0:
            return DistributionType.PS

        if len(workers) + len(chiefs) + len(masters) > 1:
            return DistributionType.MULTI_WORKER

        return DistributionType.SINGLE_WORKER

    def _patch_model(self, model, metrics, losses, loss_weights, model_call_exclude_input_index=None, squeeze=False,
                     multi_optimizer=False, use_pcgrad=False):
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
        is_distributed = isinstance(dist_strategy, (tf.distribute.experimental.MultiWorkerMirroredStrategy,
                                                    tf.distribute.experimental.ParameterServerStrategy))

        train_ds = self._create_dateset(DatasetType.TRAIN, self.train_data_args, is_distributed,
                                        global_batch_size, is_distributed)
        if train_ds is None:
            print("failed to load training data, exit, train_data_args={}".format(self.train_data_args))
            return None

        val_ds = self._create_dateset(DatasetType.VALIDATION, self.val_data_args, is_distributed,
                                      global_batch_size, is_distributed)

        if isinstance(dist_strategy, tf.distribute.experimental.ParameterServerStrategy):
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
            if tf.__version__ < TF_REF_VERSION:
                loss_scaled = tf.keras.mixed_precision.experimental.get_layer_policy(model).loss_scale is not None
            else:
                loss_scaled = tf.keras.mixed_precision.global_policy().compute_dtype == tf.float16
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

            self._patch_model(model, metrics, losses, loss_weights, model_call_exclude_input_index, input_squeeze,
                              multi_optimizer, use_pcgrad)

        st = time.perf_counter()
        print("start training model under compile-fit mode...")
        model.fit(train_ds, epochs=epochs, verbose=2, callbacks=callbacks, validation_data=val_ds,
                  validation_freq=1, steps_per_epoch=steps_per_epoch,
                  validation_steps=validation_steps if val_ds is not None else None)
        print("model training finished, cost {}s".format(time.perf_counter() - st))

        return model

    def _train_custom_loop(self, dist_strategy: tf.distribute.Strategy, global_batch_size, steps_per_epoch,
                           validation_steps):
        is_distributed = isinstance(dist_strategy, (tf.distribute.experimental.MultiWorkerMirroredStrategy,
                                                    tf.distribute.experimental.ParameterServerStrategy))

        is_ps_dist = isinstance(dist_strategy, tf.distribute.experimental.ParameterServerStrategy)

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

                if tf.__version__ < TF_REF_VERSION:
                    loss_scaled = tf.keras.mixed_precision.experimental.get_layer_policy(model).loss_scale is not None
                else:
                    loss_scaled = tf.keras.mixed_precision.global_policy().compute_dtype == tf.float16
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
                    for loss_object in loss_objects:
                        loss = loss_object(labels, predictions, sample_weights)
                        loss_vals.append(loss)
                    per_example_loss = tf.multiply(loss_vals, loss_weights)
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
            from tensorflow.python.distribute.distribute_coordinator import _WorkerContext as WC
            from tensorflow.python.distribute import distribute_coordinator_context as dc_context
            from tensorflow.python.distribute import multi_worker_util

            task_type, task_index = self._get_task_info()
            cluster_info = self._get_cluster_info()
            print("cluster_info={}, task_type='{}', task_index={}".format(cluster_info, task_type, task_index))
            worker_context = WC(
                strategy=dist_strategy,
                cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_info) if cluster_info else None,
                task_type=task_type, task_id=task_index)

            with worker_context:
                print("current_worker_context={}".format(dc_context.get_current_worker_context()))
                return __training_loop()

        return __training_loop()

    def _setup_distribute_context(self, config_args):
        has_gpu = tf.test.is_gpu_available()
        dist_type = self.get_distributed_type()
        tf.config.set_soft_device_placement(True)

        inter_op_paral = config_args.get('inter_op_paral', 0)
        intra_op_paral = config_args.get('intra_op_paral', 0)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_paral)
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_paral)
        if dist_type == DistributionType.NONE:
            if has_gpu:
                dist_strategy = tf.distribute.MirroredStrategy()
            else:
                dist_strategy = tf.distribute.get_strategy()
        else:
            if dist_type == DistributionType.PS:
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

                dist_strategy = tf.distribute.experimental.ParameterServerStrategy(
                    tf.distribute.cluster_resolver.SimpleClusterResolver(
                        tf.train.ClusterSpec(self.tf_config.get('cluster')),
                        rpc_layer='grpc'),
                    variable_partitioner=var_partitioner)
            elif dist_type == DistributionType.SINGLE_WORKER:
                dist_strategy = tf.distribute.MirroredStrategy()
            else:
                mw_com = config_args.get('mw_com', '')
                mw_com = mw_com.strip().upper()
                if mw_com == "NCCL":
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
        print("{}: train_args={}".format(__file__, self.train_args))

        dist_type = self.get_distributed_type()
        if dist_type == DistributionType.PS:
            os.environ["GRPC_FAIL_FAST"] = "use_caller"
            task_type, task_id = self._get_task_info()
            if task_type in ['worker', 'ps']:
                print("will start server for '{}' {}".format(task_type, task_id))
                server = tf.distribute.Server(tf.train.ClusterSpec(self.tf_config.get('cluster')),
                                              job_name=task_type,
                                              task_index=task_id,
                                              protocol='grpc',
                                              start=True)
                server.join()
                return

        _, dist_strategy = self._setup_distribute_context(self.train_args)
        mix_precision = self.train_args.get('mix_precision', False)
        if mix_precision:
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
        global_batch_size = batch_size * dist_strategy.num_replicas_in_sync
        print("train: dist_type={}, dist_strategy={}, dist_strategpy.extended={}, num_replicas_in_sync={},"
              " batch_size={}, global_batch_size={}".format(dist_type, dist_strategy, dist_strategy.extended,
                                                            dist_strategy.num_replicas_in_sync,
                                                            batch_size, global_batch_size))

        train_type = self.train_args.get('train_type', TRAIN_TYPE_COMPILE_FIT)
        if not train_type or not train_type.strip():
            train_type = TRAIN_TYPE_COMPILE_FIT
            print("'train_type' not set, default to {}".format(train_type))

        train_type = train_type.strip().lower()
        if train_type not in [TRAIN_TYPE_CUSTOM_LOOP, TRAIN_TYPE_COMPILE_FIT]:
            raise NotImplementedError("unknown train type '{}', supported are {}"
                                      .format(train_type, [TRAIN_TYPE_CUSTOM_LOOP, TRAIN_TYPE_COMPILE_FIT]))

        if isinstance(dist_strategy, (tf.distribute.experimental.MultiWorkerMirroredStrategy,
                                      tf.distribute.experimental.ParameterServerStrategy)):
            num_samples = self.train_args.get('num_samples')
            if not num_samples or num_samples < 0:
                raise RuntimeError("'num_samples' must be set when using distribution training")

            if num_samples < global_batch_size:
                print("WARNING: number of samples {} < num_workers*batch_size {}, auto set steps_per_epoch=1"
                      .format(num_samples, global_batch_size))
                steps_per_epoch = 1
            else:
                steps_per_epoch = num_samples // global_batch_size
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
            print("non-distributed training, ignore 'num_samples' and 'validation_steps'")
            steps_per_epoch = None
            validation_steps = None

        if train_type == TRAIN_TYPE_COMPILE_FIT:
            trained_model = self._train_compile_fit(dist_strategy, global_batch_size, steps_per_epoch, validation_steps)
        elif train_type == TRAIN_TYPE_CUSTOM_LOOP:
            trained_model = self._train_custom_loop(dist_strategy, global_batch_size, steps_per_epoch, validation_steps)

        if trained_model is not None:
            save_path = self.train_args.get("save_path")
            if not save_path or not save_path.strip():
                save_path = "saved_model"
            else:
                save_path = save_path.strip()
            save_path = make_abs_or_data_path(save_path, self.export_path, self.pack_path)
            if not self.is_chief():
                task_type, task_index = self._get_task_info()
                save_path = os.path.join(save_path, ".{}-{}_tmp".format(task_type, task_index))
            if not os.path.isdir(save_path):
                os.makedirs(save_path, exist_ok=True)
                print("created model saving dir '{}', is_chief={}".format(save_path, self.is_chief()))

            model_name = trained_model.name or split_file_name(self.user_py_file)[1]
            saved_models = []
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
            return saved_models
        return None

    def _load_model(self, model_path, model_name, purpose=None) -> tf.keras.Model:
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

    def run_evaluate(self):
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

        is_distributed = isinstance(dist_strategy, (tf.distribute.experimental.MultiWorkerMirroredStrategy,
                                                    tf.distribute.experimental.ParameterServerStrategy))

        test_ds = self._create_dateset(DatasetType.TEST, self.test_data_args, is_distributed,
                                       global_batch_size, is_distributed,
                                       check_return_type=(tf.data.Dataset, list, tuple, dict))
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

    def run_predict(self):
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
                    if row_format:
                        import re
                        headers_str = re.sub(r'\{([^{}]+)\}', lambda x: x.group(1), row_format)
                        pf.write(headers_str + '\n')
                    else:
                        pf.write(result_field_delim.join(headers) + '\n')
                    head_wrote = True

                if row_id_col:
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
