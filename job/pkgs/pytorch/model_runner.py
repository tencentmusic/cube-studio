# coding=utf-8
# @Time     : 2020/11/26 20:56
# @Auther   : lionpeng@tencent.com

import json
import os
import sys
import time
from typing import Union

import ignite
import ignite.contrib.handlers as ign_contrib_handlers
import ignite.distributed as ign_dist
import ignite.engine as ign_engine
import ignite.handlers as ign_handlers
import torch
from ignite.contrib.handlers.tensorboard_logger import global_step_from_engine, WeightsScalarHandler, \
    WeightsHistHandler, GradsScalarHandler, GradsHistHandler
from torch.utils.data import Dataset, DataLoader

from .helperfuncs import DistConfig, create_optimizer, create_loss, create_metric, try_get_model_name
from ..constants import AWFUserFunc, DatasetType, KF_UI_META_FILE
from ..utils import make_abs_or_pack_path, call_user_module, expand_param, make_abs_or_data_path, split_file_name
import shutil


class PyTorchModelRunner(object):
    def __init__(self, user_py_file, export_path, pack_path, dist_config: DistConfig = None, model_args: dict = None,
                 train_args: dict = None, evaluate_args: dict = None, train_data_args: dict = None,
                 test_data_args: dict = None, val_data_args: dict = None):
        self.user_py_file = make_abs_or_pack_path(user_py_file, export_path, pack_path)
        self.export_path = export_path
        self.pack_path = pack_path
        self.dist_config = dist_config
        self.model_args = model_args or {}
        self.train_data_args = train_data_args or {}
        self.test_data_args = test_data_args or {}
        self.val_data_args = val_data_args or {}
        self.train_args = train_args or {}
        self.evaluate_args = evaluate_args or {}
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
        if not self.dist_config:
            return True
        return not self.dist_config.world_size or self.dist_config.world_size == 1 or self.dist_config.rank == 0

    def is_distributed(self):
        if not torch.distributed.is_available():
            print("torch dose not support distrubition on this platform")
            return False
        return self.dist_config and self.dist_config.world_size is not None and self.dist_config.world_size > 1

    def _create_model(self) -> torch.nn.Module:
        model, _ = call_user_module(self.user_module, AWFUserFunc.CRETAE_MODEL, True, False, torch.nn.Module,
                                    **self.model_args)
        return ign_dist.auto_model(model)

    def _create_dateset(self, dataset_type, dataset_args, global_batch_size, drop_remainder,
                        check_return_type=Dataset) -> Union[DataLoader, list, dict, tuple]:

        if dataset_type == DatasetType.TRAIN:
            func_name = AWFUserFunc.CREATE_TRAIN_DATASET
            args = dataset_args.copy()
        elif dataset_type == DatasetType.TEST:
            func_name = AWFUserFunc.CREATE_TEST_DATASET
            args = dataset_args.copy()
        elif dataset_type == DatasetType.VALIDATION:
            func_name = AWFUserFunc.CREATE_VALIDATION_DATASET
            args = dataset_args.copy()
        else:
            raise RuntimeError("unknown dataset type '{}'".format(dataset_type))

        for k, v in args.items():
            if isinstance(v, str):
                args[k] = expand_param(v, self.export_path, self.pack_path)

        num_workers = args.get('num_workers', 4)
        num_workers = max(int(num_workers), 0)
        shuffle = args.get('shuffle', False)

        dataset, injected_args = call_user_module(self.user_module, func_name, False, True, check_return_type,
                                                  inject_args={"batch_size": global_batch_size,
                                                               "num_workers": num_workers,
                                                               "shuffle": shuffle,
                                                               "is_chief": self.is_chief()},
                                                  **args)
        if dataset is None:
            print("user function '{}' return None dataset, args={}".format(func_name, args))
            return dataset

        def __apply_options(ds):
            return ign_dist.auto_dataloader(ds, batch_size=global_batch_size, num_workers=num_workers,
                                            shuffle=shuffle, drop_last=drop_remainder)

        if isinstance(dataset, Dataset):
            return __apply_options(dataset)
        elif isinstance(dataset, (tuple, list)):
            return [__apply_options(ds) for ds in dataset]
        elif isinstance(dataset, dict):
            return {k: __apply_options(ds) for k, ds in dataset.items()}
        else:
            raise RuntimeError("user function '{}' return unsupported dataset type, args={}, only "
                               "torch.utils.data.Dataset or list/tuple/dict of torch.utils.data.Dataset are supported, "
                               "got '{}': {}".format(func_name, args, type(dataset), dataset))

    def _create_optimizer(self, model: torch.nn.Module, **kwargs):
        def __create_one_optimizer(trainable_vars, optimizer_detail):
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
                        args = optimizer_detail.get('args', {})
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
            optimizer = create_optimizer(trainable_vars, optimizer_type, lr, **args)
            if optimizer is None:
                raise NotImplementedError("unsupported optimizer type '{}'".format(optimizer_type))

            print("create optimizer of type '{}' of learning rate {}, args={}".format(optimizer_type, lr, args))
            return ign_dist.auto_optim(optimizer)

        optimizer_specs = self.train_args.get('optimizer')
        if not isinstance(optimizer_specs, (list, tuple)):
            optimizer_specs = [optimizer_specs]
        if len(optimizer_specs) > 1:
            print("user specified multiple optimizers: {}".format(optimizer_specs))
            trainable_var_groups, _ = call_user_module(self.user_module, AWFUserFunc.GROUP_TRAINABLE_VARS,
                                                       True, False, (list, tuple), model=model)
            if len(trainable_var_groups) != len(optimizer_specs):
                raise RuntimeError("#optimizers {} != #trainable variable groups {}"
                                   .format(len(optimizer_specs), len(trainable_var_groups)))

            return [__create_one_optimizer(var_group, opt_spec) for var_group, opt_spec in
                    zip(trainable_var_groups, optimizer_specs)]

        assert model is not None
        return __create_one_optimizer(model.parameters(), optimizer_specs[0])

    def _create_losses(self, is_training=True, **kwargs):
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

        def __parse_single_loss(detail):
            loss, loss_weight = None, None
            if isinstance(detail, str):
                loss_type = detail.strip()
                if not loss_type:
                    raise RuntimeError("loss can not be empty string, is_training={}".format(is_training))
                args = kwargs or {}
                loss = create_loss(loss_type, **args)
                if loss is None:
                    raise NotImplementedError("unsupported loss type '{}', is_training={}"
                                              .format(loss_type, is_training))
                print("create '{}' loss {} by str '{}', is_training={}".format(loss_type, loss, detail, is_training))
            else:
                loss_type = detail.get('type')
                if not loss_type or not loss_type.strip():
                    raise RuntimeError("loss type not set, is_training={}".format(is_training))
                loss_type = loss_type.strip()
                args = detail.get('args', {})
                if kwargs:
                    args.update(kwargs)
                loss = create_loss(loss_type, **args)
                if loss is None:
                    raise NotImplementedError("unsupported loss type '{}', is_training={}"
                                              .format(loss_type, is_training))
                loss_weight = detail.get('weight')
                if loss_weight is not None:
                    loss_weight = float(loss_weight)
                print("create '{}' loss {} by dict '{}' with weight {}, is_training={}"
                      .format(loss_type, loss, detail, loss_weight, is_training))
            loss = loss.to(ign_dist.device())
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
            weights = [1.]*len(losses)
        elif len(losses) == 1:
            losses = losses[0]
            weights = None
        return losses, weights

    def _create_metrics(self, name_prefix=None, is_training=True, **kwargs):
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

        def __parse_single_metric(detail, output_idx=None):
            if isinstance(detail, str):
                metric_type = detail.strip()
                if not metrics_detail:
                    raise RuntimeError("metric can not be empty string, is_training={}".format(is_training))
                args = kwargs or {}
                if output_idx is not None:
                    args['output_transform'] = lambda pred_y, y: (pred_y[output_idx], y[output_idx])
                metric = create_metric(metric_type, **args)
                if metric is None:
                    raise NotImplementedError("unsupported metric type '{}', is_training={}"
                                              .format(metric_type, is_training))
                name = metric_type if name_prefix is None else str(name_prefix) + '_' + metric_type
                print("create '{}' '{}' metric {} by str '{}', is_training={}"
                      .format(name_prefix, metric_type, metric, detail, is_training))
            else:
                metric_type = detail.get('type')
                if not metric_type or not metric_type.strip():
                    raise RuntimeError("metric type not set, is_training={}".format(is_training))
                metric_type = metric_type.strip()
                args = detail.get('args', {})
                if kwargs:
                    args.update(kwargs)
                if output_idx is not None:
                    args['output_transform'] = lambda pred_y, y: (pred_y[output_idx], y[output_idx])
                metric = create_metric(metric_type, **args)
                if metric is None:
                    raise NotImplementedError("unsupported metric type '{}', is_training={}"
                                              .format(metric_type, is_training))
                name = metric_type if name_prefix is None else str(name_prefix) + '_' + metric_type
                print("create '{}' '{}' metric {} by dict '{}', is_training={}"
                      .format(name_prefix, metric_type, metric, detail, is_training))
            if output_idx is not None:
                name += '_' + str(output_idx)
            return name, metric

        metric_list = []
        if isinstance(metrics_detail, (str, dict)):
            n, m = __parse_single_metric(metrics_detail)
            metric_list.append({n: m})
        else:
            multi_output = len(metrics_detail) > 1
            for i, metric_item in enumerate(metrics_detail):
                if not isinstance(metric_item, (str, dict, list)):
                    raise RuntimeError("{}th metric must be str or dict or list, got {}, is_training={}"
                                       .format(i, metric_item, is_training))
                index = i if multi_output else None
                if isinstance(metric_item, (str, dict)):
                    n, m = __parse_single_metric(metric_item, index)
                    metric_list.append({n: m})
                else:
                    sub_metric_list = {}
                    for j, sub_item in enumerate(metric_item):
                        if not isinstance(sub_item, (str, dict)):
                            raise RuntimeError("({}, {})th metric must be str or dict, got {}, is_training={}"
                                               .format(i, j, sub_item, is_training))
                        sub_n, sub_m = __parse_single_metric(sub_item, index)
                        sub_metric_list[sub_n] = sub_m
                    metric_list.append(sub_metric_list)

        print("created metrics={}, name_prefix='{}', is_training={}".format(metric_list, name_prefix, is_training))
        return metric_list

    def _create_callbacks(self, trainer: ign_engine.Engine, trainer_evaluator: ign_engine.Engine,
                          val_evaluator: ign_engine.Engine, model: torch.nn.Module, optimizers):
        callbacks = []
        if self.is_chief():
            ckpt_path = self.train_args.get('ckpt_path')
            if not ckpt_path or not ckpt_path.strip():
                ckpt_path = 'checkpoints'
            else:
                ckpt_path = ckpt_path.strip()
            ckpt_path = make_abs_or_data_path(ckpt_path, self.export_path, self.pack_path)
            if os.path.isdir(ckpt_path):
                shutil.rmtree(ckpt_path, ignore_errors=True)
                print("deleted previous existed check point dir '{}'".format(ckpt_path))
            os.makedirs(ckpt_path, exist_ok=True)
            print("created check point dir '{}'".format(ckpt_path))
            ckpt_saver = ign_handlers.DiskSaver(ckpt_path, require_empty=False)
            ckpt_to_save = {"model": model}
            if isinstance(optimizers, (list, tuple)):
                if len(optimizers) > 1:
                    for i, opt in enumerate(optimizers):
                        ckpt_to_save["opt_%s" % i] = opt
                else:
                    ckpt_to_save["optimizer"] = optimizers[0]
            else:
                ckpt_to_save["optimizer"] = optimizers

            def ckpt_score_fn(eng):
                return -val_evaluator.state.metrics["val_loss"]

            ckpt_callback = ign_handlers.Checkpoint(
                ckpt_to_save,
                ckpt_saver,
                score_function=ckpt_score_fn if val_evaluator is not None else None,
                score_name="val_loss" if val_evaluator is not None else None,
                n_saved=1)
            trainer.add_event_handler(ign_engine.Events.EPOCH_COMPLETED, ckpt_callback)
            callbacks.append(ckpt_callback)
            print("added checkpoint callback, checkpoint dir='{}'".format(ckpt_path))

        early_stop_spec = self.train_args.get("early_stopping")
        if early_stop_spec is not None and val_evaluator is not None:
            metric_name = early_stop_spec.get('monitor', '').strip()
            if not metric_name:
                raise RuntimeError("'monitor' not set in 'early_stopping' settings")
            min_delta = early_stop_spec.get('min_delta', 0)
            if min_delta < 0:
                raise RuntimeError("'min_delta' in 'early_stopping' settings must be >= 0, get {}".format(min_delta))
            patience = early_stop_spec.get('patience', 5)
            if patience < 1:
                raise RuntimeError("'patience' in 'early_stopping' settings must be >= 1, get {}".format(patience))
            mode = early_stop_spec.get('mode', 'max').strip().lower()
            if mode not in ['max', 'min']:
                raise RuntimeError("'mode' in 'early_stopping' settings must be 'max' or 'min', got '{}'".format(mode))

            def es_score_fn(eng):
                return eng.state.metrics[metric_name] if mode == 'max' else -eng.state.metrics[metric_name]

            es_callback = ign_handlers.EarlyStopping(patience, es_score_fn, trainer, min_delta)
            val_evaluator.add_event_handler(ign_engine.Events.EPOCH_COMPLETED, es_callback)
            callbacks.append(es_callback)
            print("added early stopping callback, spec={}".format(early_stop_spec))
        elif val_evaluator is None:
            print("WARNING: val_evaluator is not None, can not setup early stopping callback, please check if"
                  " user function '{}' is avaliable".format(AWFUserFunc.CREATE_TEST_DATASET))

        trainspeed_log_spec = self.train_args.get("train_speed_logger")
        if trainspeed_log_spec is not None:
            from .extend_callbacks import TrainSpeedLoggerCallback
            tsl_callback = TrainSpeedLoggerCallback(**trainspeed_log_spec)
            tsl_callback.attach(trainer)
            callbacks.append(tsl_callback)
            print("added train speed logger callback, spec={}".format(trainspeed_log_spec))

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
                tb_callback = ign_contrib_handlers.TensorboardLogger(log_dir=summary_path)
                tb_callback.attach_output_handler(trainer,
                                                  event_name=ign_engine.Events.EPOCH_COMPLETED,
                                                  tag='training',
                                                  metric_names='all')
                tb_callback.attach_output_handler(trainer_evaluator,
                                                  event_name=ign_engine.Events.EPOCH_COMPLETED,
                                                  tag='training',
                                                  metric_names='all',
                                                  global_step_transform=global_step_from_engine(trainer))
                if val_evaluator is not None:
                    tb_callback.attach_output_handler(val_evaluator,
                                                      event_name=ign_engine.Events.EPOCH_COMPLETED,
                                                      tag='validation',
                                                      metric_names='all',
                                                      global_step_transform=global_step_from_engine(trainer))
                weights_log_rate = tensorboard_spec.get('weights', '1e').strip().lower()
                if weights_log_rate:
                    if weights_log_rate.endswith('b'):
                        every = int(weights_log_rate[:-1])
                        event = ign_engine.Events.ITERATION_COMPLETED(every=every)
                    elif weights_log_rate.endswith('e'):
                        every = int(weights_log_rate[:-1])
                        event = ign_engine.Events.EPOCH_COMPLETED(every=every)
                    else:
                        raise RuntimeError("'weights' in 'tensorboard' setting should be end with 'b' or 'e',"
                                           " got '{}'".format(weights_log_rate))
                    tb_callback.attach(trainer, log_handler=WeightsScalarHandler(model), event_name=event)
                    tb_callback.attach(trainer, log_handler=WeightsHistHandler(model), event_name=event)
                    print("added weights loggers in tensorboard")

                grads_log_rate = tensorboard_spec.get('grads', '1e').strip().lower()
                if grads_log_rate:
                    if grads_log_rate.endswith('b'):
                        every = int(grads_log_rate[:-1])
                        event = ign_engine.Events.ITERATION_COMPLETED(every=every)
                    elif grads_log_rate.endswith('e'):
                        every = int(grads_log_rate[:-1])
                        event = ign_engine.Events.EPOCH_COMPLETED(every=every)
                    else:
                        raise RuntimeError("'grads' in 'tensorboard' setting should be end with 'b' or 'e',"
                                           " got '{}'".format(grads_log_rate))
                    tb_callback.attach(trainer, log_handler=GradsScalarHandler(model), event_name=event)
                    tb_callback.attach(trainer, log_handler=GradsHistHandler(model), event_name=event)
                    print("added grads loggers in tensorboard")

                optimizers_log_rate = tensorboard_spec.get('optimizer', '1e').strip().lower()
                if optimizers_log_rate:
                    if optimizers_log_rate.endswith('b'):
                        every = int(optimizers_log_rate[:-1])
                        event = ign_engine.Events.ITERATION_COMPLETED(every=every)
                    elif optimizers_log_rate.endswith('e'):
                        every = int(optimizers_log_rate[:-1])
                        event = ign_engine.Events.EPOCH_COMPLETED(every=every)
                    else:
                        raise RuntimeError("'optimizer' in 'tensorboard' setting should be end with 'b' or 'e',"
                                           " got '{}'".format(grads_log_rate))
                    if not isinstance(optimizers, (tuple, list)):
                        optimizers = [optimizers]
                    for optimizer in optimizers:
                        tb_callback.attach_opt_params_handler(trainer, event_name=event, optimizer=optimizer)
                    print("added optimizer loggers in tensorboard")
                callbacks.append(tb_callback)
        return callbacks

    def _model_to_save(self, model: torch.nn.Module) -> torch.nn.Module:
        trained_model = model
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            trained_model = model.module
            print("original model is a parallized model {}, extracted target model {}"
                  .format(type(model), type(trained_model)))
        model_to_save, _ = call_user_module(self.user_module, AWFUserFunc.CREATE_MODEL_TO_SAVE, False, True,
                                            (torch.nn.Module, tuple), trained_model=trained_model)
        save_options = {}
        if model_to_save is not None:
            if isinstance(model_to_save, tuple):
                if len(model_to_save) == 0:
                    print("user function '{}' return no model info".format(AWFUserFunc.CREATE_MODEL_TO_SAVE))
                    final_model = model
                elif len(model_to_save) > 2:
                    raise RuntimeError("user function '{}' should return <Model> or (<Model>, <SaveOption>), got {}"
                                       .format(AWFUserFunc.CREATE_MODEL_TO_SAVE, model_to_save))
                else:
                    final_model = model_to_save[0]
                    if len(model_to_save) > 1:
                        save_options = model_to_save[1]
            else:
                final_model = model_to_save
        else:
            final_model = trained_model

        return final_model, save_options

    @classmethod
    def _extract_batch_data(cls, batch):
        if not isinstance(batch, (tuple, list)):
            raise RuntimeError("batch should be or tuple/list of (x, y) or (x, y, w), got {}".format(batch))
        x, ys, ws = None, None, None
        if len(batch) == 2:
            x, ys = batch
        elif len(batch) == 3:
            x, ys, ws = batch
            if isinstance(ys, (tuple, list)):
                if not isinstance(ws, (tuple, list)) or len(ys) != len(ws):
                    raise RuntimeError("#lables {} != #sample weights {}, got x={}, ys={}, ws={}"
                                       .format(len(ys), len(ws) if isinstance(ws, (tuple, list)) else 1),
                                       x, ys, ws)
        else:
            raise RuntimeError("batch should be a tuple/list of len 2~3, means x,y,w(optional). got {}"
                               .format(batch))
        if x is None or ys is None:
            raise RuntimeError("x and ys in batch must not be None, got x={}, ys={}".format(x, ys))
        return x, ys, ws

    def _create_evaluator_engine(self, model: torch.nn.Module, metrics, loss_name_prefix=None,
                                 is_training=True) -> ign_engine.Engine:
        def evaluation_step(engine, batch):
            x, ys, _ = self._extract_batch_data(batch)
            if not isinstance(ys, (tuple, list)):
                ys = [ys]
            if x.device != ign_dist.device():
                x = x.to(ign_dist.device(), non_blocking=True)
                ys = [y.to(ign_dist.device(), non_blocking=True) for y in ys]

            model.eval()
            with torch.no_grad():
                y_preds = model(x)
                if not isinstance(y_preds, (tuple, list)):
                    y_preds = [y_preds]
            if len(ys) != len(y_preds):
                raise RuntimeError("#outputs {}, #labels {} mismatch during evaluation, got y_preds={}, ys={}"
                                   .format(len(y_preds), len(ys), y_preds, ys))
            if len(y_preds) > 1:
                return y_preds, ys
            return y_preds[0], ys[0]

        evaluator = ign_engine.Engine(evaluation_step)
        if not isinstance(metrics, (tuple, list)):
            metrics = [metrics]
        for ms in metrics:
            for n, m in ms.items():
                m.attach(evaluator, n)
                print("attached metric '{}' {} to evaluation engine {}".format(n, m, evaluator))

        loss_objects, loss_weights = self._create_losses(is_training, reduction='mean')
        if loss_objects is not None:
            if not isinstance(loss_objects, (tuple, list)):
                loss_objects = [loss_objects]
            if not isinstance(loss_weights, (tuple, list)):
                if loss_weights is not None:
                    loss_weights = [loss_weights]
                else:
                    loss_weights = [None] * len(loss_objects)

            if loss_objects:
                multi_output = len(loss_objects) > 1
                total_loss = 0
                for i, (loss_obj, loss_w) in enumerate(zip(loss_objects, loss_weights)):
                    loss_m = ignite.metrics.Loss(loss_obj)
                    loss_n = "loss_{}".format(i) if multi_output else "loss"
                    if loss_name_prefix:
                        loss_n = str(loss_name_prefix) + '_' + loss_n
                    loss_m.attach(evaluator, loss_n)
                    print("attached loss metric '{}' {} to evaluation engine {}".format(loss_n, loss_m, evaluator))
                    if multi_output:
                        total_loss += loss_m * loss_w if loss_w is not None else loss_m

                if multi_output:
                    total_loss_m = ignite.metrics.Loss(total_loss)
                    total_loss_n = str(loss_name_prefix) + "_total_loss" if loss_name_prefix else "total_loss"
                    total_loss_m.attach(evaluator, total_loss_n)
                    print("attached total loss metric '{}' {} to evaluation engine {}"
                          .format(total_loss_n, total_loss_m, evaluator))

        return evaluator

    def _train(self, global_batch_size, parallel_context: ign_dist.Parallel):
        train_ds = self._create_dateset(DatasetType.TRAIN, self.train_data_args, global_batch_size, True)
        if train_ds is None:
            print("failed to load training data, exit, train_data_args={}".format(self.train_data_args))
            return None
        test_ds = self._create_dateset(DatasetType.TEST, self.test_data_args, global_batch_size, True)

        epochs = self.train_args.get('epochs')
        if not epochs or epochs < 0:
            epochs = 1
            print("epochs not properly set, default to 1")

        def __training_loop(local_rank, model: torch.nn.Module):
            print("training loop: local_rank={}, model={}".format(local_rank, model))
            optimizers = self._create_optimizer(model)
            if not isinstance(optimizers, (tuple, list)):
                optimizers = [optimizers]
            loss_objects, loss_weights = self._create_losses(True, reduction='none')
            if not isinstance(loss_objects, (tuple, list)):
                loss_objects = [loss_objects]
            if not isinstance(loss_weights, (tuple, list)):
                if loss_weights is not None:
                    loss_weights = [loss_weights]
                else:
                    loss_weights = [None] * len(loss_objects)

            def train_step(engine, batch):
                x, ys, ws = self._extract_batch_data(batch)
                if not isinstance(ys, (tuple, list)):
                    ys = [ys]
                if ws is not None and not isinstance(ws, (tuple, list)):
                    ws = [ws]
                elif ws is None:
                    ws = [None]*len(ys)
                if x.device != ign_dist.device():
                    x = x.to(ign_dist.device(), non_blocking=True)
                    ys = [y.to(ign_dist.device(), non_blocking=True) for y in ys]
                    ws = [w.to(ign_dist.device(), non_blocking=True) if w is not None else None for w in ws]

                model.train()
                y_preds = model(x)
                if not isinstance(y_preds, (tuple, list)):
                    y_preds = [y_preds]
                if len(y_preds) != len(loss_objects) or len(y_preds) != len(ys):
                    raise RuntimeError("#outputs {}, #labels {}, #losses {} mismatch during training, got y_preds={},"
                                       " ys={}, losses={}".format(len(y_preds), len(ys), len(loss_objects), y_preds,
                                                                  ys, loss_objects))
                loss_vals = []
                total_loss = 0
                for y_p, y, sample_w, loss_obj, loss_w in zip(y_preds, ys, ws, loss_objects, loss_weights):
                    loss = loss_obj(y_p, y)
                    if sample_w is not None:
                        loss = loss * sample_w
                    loss = torch.mean(loss)
                    if loss_w is not None:
                        loss *= loss_w
                    loss_vals.append(loss)
                    total_loss += loss

                for optimizer in optimizers:
                    optimizer.zero_grad()
                total_loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

                return {"losses": [loss.item() for loss in loss_vals], "total_loss": total_loss.item()}

            train_metrics = self._create_metrics("train", True)
            trainer = ign_engine.Engine(train_step)
            train_evaluator = self._create_evaluator_engine(model, train_metrics, is_training=True)
            if test_ds is not None:
                test_metrics = self._create_metrics("val", False)
                val_evaluator = self._create_evaluator_engine(model, test_metrics, "val", True)
            else:
                val_evaluator = None

            def __process_log(engine):
                epoch_num = trainer.state.epoch
                max_epochs = trainer.state.max_epochs
                train_eval_state = train_evaluator.run(train_ds)
                msg = "epoch {}/{}: train time {}s, train-eval time {}s, train-metrics={}"\
                    .format(epoch_num, max_epochs, trainer.state.times["EPOCH_COMPLETED"],
                            train_eval_state.times["EPOCH_COMPLETED"],
                            train_eval_state.metrics)
                if test_ds is not None:
                    val_eval_state = val_evaluator.run(test_ds)
                    msg += ", val-eval time {}s, val-metrics={}".format(val_eval_state.times["EPOCH_COMPLETED"],
                                                                        val_eval_state.metrics)
                print(msg)

            trainer.add_event_handler(ign_engine.Events.EPOCH_COMPLETED, __process_log)
            self._create_callbacks(trainer, train_evaluator, val_evaluator, model, optimizers)
            trainer.run(train_ds, max_epochs=epochs)

        st = time.perf_counter()
        print("start training, distribution config={}".format(self.dist_config))
        model = self._create_model()
        parallel_context.run(__training_loop, model)
        print("training finished, cost {}s".format(time.perf_counter()-st))

        return model

    def _restore_best_model_from_checkpoint(self, model: torch.nn.Module):
        ckpt_path = self.train_args.get('ckpt_path', 'checkpoints').strip()
        ckpt_path = make_abs_or_data_path(ckpt_path, self.export_path, self.pack_path)
        if not os.path.isdir(ckpt_path):
            print("WARNING: checkpoint path '{}' not exists, can not restore model parameters".format(ckpt_path))
            return False

        import glob
        ckpt_files = glob.glob(os.path.join(ckpt_path, "*"))
        if not ckpt_files:
            print("WARNING: found checkpoint files in path '{}', can not restore model parameters".format(ckpt_path))
            return False
        if len(ckpt_files) == 1:
            best_ckpt_file = ckpt_files[0]
        else:
            def sort_key_fn(fname):
                _, basename, _ = split_file_name(fname)
                score = basename.split('=')[-1]
                return float(score)

            sorted(ckpt_files, key=sort_key_fn)
            best_ckpt_file = ckpt_files[-1]

        print("found best checkpoint file '{}' during {}".format(best_ckpt_file, ckpt_files))
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module
        checkpoint = torch.load(best_ckpt_file)
        if not checkpoint:
            print("WARNING: loaded empty data from checkpoint file '{}', can not restore model parameters"
                  .format(best_ckpt_file))
            return
        if 'model' not in checkpoint:
            print("found no 'model' key in checkpoint loaded from '{}', got {}, assume it's wholly model"
                  .format(best_ckpt_file, checkpoint))
            model.load_state_dict(checkpoint)
            print("restored model parameters from whole checkpoint file '{}'".format(best_ckpt_file))
        else:
            model.load_state_dict(checkpoint['model'])
            print("restored model parameters from 'model' key of checkpoint file '{}'".format(best_ckpt_file))
        return True

    def run_train(self):
        print("{}: train_args={}".format(__file__, self.train_args))

        backend = None
        if self.is_distributed():
            backend = self.train_args.get('mw_com', 'GLOO').strip().lower()
            if backend == "nccl" and (not torch.distributed.is_nccl_available() or not torch.cuda.is_available()):
                print("WARNING: nccl or cuda is not available, fallback to gloo")
                backend = "gloo"

        batch_size = self.train_args.get('batch_size', 128)
        if batch_size <= 0:
            print("'batch_size' inproperly set to {}, changed to 128".format(batch_size))
            batch_size = 128
        g_batch_size = batch_size * self.dist_config.world_size if self.is_distributed() else batch_size
        if self.is_distributed():
            if self.dist_config.local_rank is None:
                os.environ['LOCAL_RANK'] = '0'
                print("'LOCAL_RANK' not set in envs, manually set it to 0")
            parallel_ctx = ign_dist.Parallel(backend, nnodes=self.dist_config.world_size,
                                             node_rank=self.dist_config.rank,
                                             master_addr=self.dist_config.master_addr,
                                             master_port=self.dist_config.master_port)
        else:
            parallel_ctx = ign_dist.Parallel(None)
        print("backend={}, batch_size={}, global_batch_size={}, parallel_ctx={}"
              .format(backend, batch_size, g_batch_size, parallel_ctx))

        with parallel_ctx:
            trained_model = self._train(g_batch_size, parallel_ctx)
            if trained_model is not None:
                save_path = self.train_args.get("save_path")
                if not save_path or not save_path.strip():
                    save_path = "saved_model-{}.pt".format(time.strftime('%Y%m%d%H%M%S'))
                else:
                    save_path = save_path.strip()
                save_path = make_abs_or_data_path(save_path, self.export_path, self.pack_path)
                model_name = try_get_model_name(trained_model)
                if self.is_chief():
                    try:
                        self._restore_best_model_from_checkpoint(trained_model)
                    except Exception as e:
                        import traceback
                        print("WARING: failed to restore best model parameters from checkpoint: {}\n{}"
                              .format(e, traceback.format_exc()))
                    model_to_save, save_options = self._model_to_save(trained_model)
                    torch.save(model_to_save, save_path, **save_options)
                    print("saved model({}) to '{}'".format(type(model_to_save), save_path))
                    model_name = try_get_model_name(model_to_save)
                return save_path, model_name
        return None, None

    def _load_model(self, model_path, model_name) -> torch.nn.Module:
        model, _ = call_user_module(self.user_module, AWFUserFunc.LOAD_MODEL, False, True, torch.nn.Module,
                                    path=model_path, name=model_name)
        if not model:
            print("user function '{}' did not return a model, will try to load it from '{}'"
                  .format(AWFUserFunc.LOAD_MODEL, model_path))
            model = torch.load(model_path, map_location=torch.device('cpu'))
            print("loaded model '{}' from '{}': {}".format(model_name, model_path, model))
        else:
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
        batch_size = self.evaluate_args.get('batch_size', 128)
        if batch_size <= 0:
            print("evaluation 'batch_size' inproperly set to {}, changed to 128".format(batch_size))
            batch_size = 128
        val_ds = self._create_dateset(DatasetType.VALIDATION, self.val_data_args, batch_size, False,
                                      (Dataset, list, tuple, dict))
        if not val_ds:
            print("got no validation data, exit")
            return None

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
            model_name = spec.get('name', '').strip()
            if not model_name:
                default_name = 'model_%s' % i
                model_name = self.model_args.get('name', default_name) if self.model_args else default_name
            model = self._load_model(model_path, model_name)
            if model is None:
                print("failed to load model '{}' from '{}'".format(model_name, model_path))
                continue
            model_name = try_get_model_name(model) or model_name
            metrics = self._create_metrics(is_training=False)
            evaluator = self._create_evaluator_engine(model, metrics, is_training=False)
            loaded_models.append((model_name, model_path, evaluator))

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
        for i, (name, path, evaluator) in enumerate(loaded_models):
            st = time.perf_counter()
            if isinstance(val_ds, DataLoader):
                state = evaluator.run(val_ds)
                eval_result = state.metrics
                eval_results.append((name, path, __reconstruct_evaluation_result(eval_result)))
            elif isinstance(val_ds, (list, tuple)):
                eval_result = {}
                for idx, ds in enumerate(val_ds):
                    state = evaluator.run(val_ds)
                    eval_result_per_ds = state.metrics
                    eval_result.update(__reconstruct_evaluation_result(eval_result_per_ds, idx))
                eval_results.append((name, path, eval_result))
            else:
                eval_result = {}
                for n, ds in val_ds.items():
                    state = evaluator.run(val_ds)
                    eval_result_per_ds = state.metrics
                    eval_result.update(__reconstruct_evaluation_result(eval_result_per_ds, n))
                eval_results.append((name, path, eval_result))
            print("evaluated {}th model '{}' from path '{}', cost {}s: {}"
                  .format(i, name, path, time.perf_counter()-st, eval_result))
        return eval_results
