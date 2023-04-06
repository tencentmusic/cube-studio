# coding=utf-8
# @Time     : 2020/11/27 15:32
# @Auther   : lionpeng@tencent.com
import os
import torch
import ignite
from typing import Union


class DistConfig(object):
    __slots__ = ['world_size', 'rank', 'master_port', 'master_addr', 'local_rank']

    def __str__(self):
        d = {a: getattr(self, a) for a in self.__slots__}
        return str(d)

    @classmethod
    def from_env(cls):
        cfg = DistConfig()
        for a in cls.__slots__:
            v = os.environ.get(a.upper())
            if a in ['world_size', 'rank', 'master_port'] and v is not None:
                v = int(v)
            cfg.__setattr__(a, v)
        return cfg


def create_optimizer(trainable_vars, optimizer_type, lr, **kwargs):
    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adam":
        return torch.optim.Adam(trainable_vars, lr=lr, **kwargs)
    if optimizer_type == "adamax":
        return torch.optim.Adamax(trainable_vars, lr=lr, **kwargs)
    if optimizer_type == "adadelta":
        return torch.optim.Adadelta(trainable_vars, lr=lr, **kwargs)
    if optimizer_type == "adagrad":
        return torch.optim.Adagrad(trainable_vars, lr=lr, **kwargs)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(trainable_vars, lr=lr, **kwargs)
    if optimizer_type == "rmsprop":
        return torch.optim.RMSprop(trainable_vars, lr=lr, **kwargs)
    if optimizer_type == "sgd":
        return torch.optim.SGD(trainable_vars, lr=lr, **kwargs)
    if optimizer_type == "sparseadam":
        return torch.optim.SparseAdam(trainable_vars, lr=lr, **kwargs)
    if optimizer_type == "asgd":
        return torch.optim.ASGD(trainable_vars, lr=lr, **kwargs)
    if optimizer_type == "rprop":
        return torch.optim.Rprop(trainable_vars, lr=lr, **kwargs)
    return None


def create_loss(loss_type, **kwargs):
    loss_type = loss_type.lower()

    if loss_type in ['binary_cross_entropy', 'binary_crossentropy', 'bin_cross_entropy', 'bin_crossentropy', 'bce']:
        if kwargs.pop('with_logits', False):
            return torch.nn.BCEWithLogitsLoss(**kwargs)
        return torch.nn.BCELoss(**kwargs)
    if loss_type in ['categorical_cross_entropy', 'categorical_crossentropy', 'cate_cross_entropy',
                     'cate_crossentropy', 'cce']:
        return torch.nn.CrossEntropyLoss(**kwargs)
    if loss_type in ['categorical_hinge', 'catehinge', 'cate_hinge', 'chinge', 'ch']:
        return torch.nn.MultiMarginLoss(**kwargs)
    if loss_type in ['huber', 'sl1']:
        return torch.nn.SmoothL1Loss(**kwargs)
    if loss_type in ['kullback_leibler_divergence', 'kld']:
        return torch.nn.KLDivLoss(**kwargs)
    if loss_type in ['mean_absolute_error', 'mean_abs_error', 'mae', 'l1']:
        return torch.nn.L1Loss(**kwargs)
    if loss_type in ['mean_squared_error', 'mean_square_error', 'mse']:
        return torch.nn.MSELoss(**kwargs)
    if loss_type in ['ctc']:
        return torch.nn.CTCLoss(**kwargs)
    if loss_type in ['nll', 'neg_ll', 'neg_log_likelihood', 'negative_log_likelihood']:
        return torch.nn.NLLLoss(**kwargs)
    if loss_type in ['pnll', 'p_neg_ll', 'p_neg_log_likelihood', 'poisson_negative_log_likelihood']:
        return torch.nn.PoissonNLLLoss(**kwargs)
    return None


def create_metric(metric_type, **kwargs):
    metric_type = metric_type.lower()

    if metric_type in ['accuracy', 'acc']:
        return ignite.metrics.Accuracy(**kwargs)
    if metric_type in ['auc']:
        from ignite.contrib.metrics import ROC_AUC
        return ROC_AUC(**kwargs)
    if metric_type in ['binary_accuracy', 'binary_acc', 'bin_acc', 'bacc']:
        kwargs["is_multilabel"] = False
        return ignite.metrics.Accuracy(**kwargs)
    if metric_type in ['binary_cross_entropy', 'binary_crossentropy', 'bin_cross_entropy', 'bin_crossentropy', 'bce']:
        if kwargs.pop('with_logits', False):
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            loss = torch.nn.BCELoss()
        kwargs["loss_fn"] = loss
        return ignite.metrics.Loss(**kwargs)
    if metric_type in ['categorical_accuracy', 'categorical_acc', 'cate_accuracy', 'cate_acc', 'cacc']:
        return ignite.metrics.Accuracy(**kwargs)
    if metric_type in ['categorical_cross_entropy', 'categorical_crossentropy', 'cate_cross_entropy',
                       'cate_crossentropy', 'cce']:
        loss = torch.nn.CrossEntropyLoss()
        kwargs["loss_fn"] = loss
        return ignite.metrics.Loss(**kwargs)
    if metric_type in ['categorical_hinge', 'catehinge', 'cate_hinge', 'chinge', 'ch']:
        loss = torch.nn.MultiMarginLoss()
        kwargs["loss_fn"] = loss
        return ignite.metrics.Loss(**kwargs)
    if metric_type in ['kullback_leibler_divergence', 'kld']:
        loss = torch.nn.KLDivLoss()
        kwargs["loss_fn"] = loss
        return ignite.metrics.Loss(**kwargs)
    if metric_type in ['mean_absolute_error', 'mean_abs_error', 'mae']:
        return ignite.metrics.MeanAbsoluteError(**kwargs)
    if metric_type in ['mean_squared_error', 'mean_square_error', 'mse']:
        return ignite.metrics.MeanSquaredError(**kwargs)
    if metric_type in ['precision', 'prec']:
        return ignite.metrics.Precision(**kwargs)
    if metric_type in ['recall']:
        return ignite.metrics.Recall(**kwargs)
    if metric_type in ['root_mean_squared_error', 'root_mean_square_error', 'rmse']:
        return ignite.metrics.RootMeanSquaredError(**kwargs)
    if metric_type in ['top_k_categorical_accuracy', 'top_k_categorical_acc', 'top_k_cate_accuracy',
                       'top_k_cate_acc', 'top_k_cacc']:
        return ignite.metrics.TopKCategoricalAccuracy(**kwargs)
    return None


def try_get_model_name(model: Union[torch.nn.Module, str]):
    if model is None:
        return None
    loaded_model = None
    if isinstance(model, str):
        model_path = model
        if os.path.isfile(model_path):
            print("model file '{}' not exists, can not get model name from it")
            return None
        try:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            loaded_model = model
        except Exception as e:
            import traceback
            print("load model from file '{}' error: {}\n{}".format(model_path, e, traceback.format_exc()))
            return None
    if isinstance(model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = getattr(model, 'module', model)
    if hasattr(model, 'name'):
        name = getattr(model, 'name')
        name = str(name).strip()
    if loaded_model is not None:
        del loaded_model
    return name or type(model).__name__
