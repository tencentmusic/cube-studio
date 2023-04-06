# coding=utf-8

import os

import tensorflow as tf

from .extend_activations import Dice
from .extend_utils import PCGrad

TF_REF_VERSION = '2.6.0'


def create_optimizer(optimizer_type, lr, v1=False, **kwargs):
    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adam":
        if v1:
            return tf.compat.v1.train.AdamOptimizer(learning_rate=lr, **kwargs)
        return tf.keras.optimizers.Adam(learning_rate=lr, **kwargs)
    if optimizer_type == "adamax":
        if v1:
            return None
        return tf.keras.optimizers.Adamax(learning_rate=lr, **kwargs)
    if optimizer_type == "adadelta":
        if v1:
            return tf.compat.v1.train.AdadeltaOptimizer(learning_rate=lr, **kwargs)
        return tf.keras.optimizers.Adadelta(learning_rate=lr, **kwargs)
    if optimizer_type == "adagrad":
        if v1:
            return tf.compat.v1.train.AdagradOptimizer(learning_rate=lr, **kwargs)
        return tf.keras.optimizers.Adagrad(learning_rate=lr, **kwargs)
    if optimizer_type == "nadam":
        if v1:
            return None
        return tf.keras.optimizers.Nadam(learning_rate=lr, **kwargs)
    if optimizer_type == "rmsprop":
        if v1:
            return tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr, **kwargs)
        return tf.keras.optimizers.RMSprop(learning_rate=lr, **kwargs)
    if optimizer_type == "sgd":
        if v1:
            return tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr, **kwargs)
        return tf.keras.optimizers.SGD(learning_rate=lr, **kwargs)
    if optimizer_type == "ftrl":
        if v1:
            return tf.compat.v1.train.FtrlOptimizer(learning_rate=lr, **kwargs)
        return tf.keras.optimizers.Ftrl(learning_rate=lr, **kwargs)
    if optimizer_type in ["padagrad", "proximaladagrad"]:
        return tf.compat.v1.train.ProximalAdagradOptimizer(learning_rate=lr, **kwargs)
    return None


def create_grad_process(grad_process):
    if grad_process is None:
        return None
    grad_process = grad_process.lower()
    if grad_process == 'pcgrad':
        return PCGrad
    return None


def create_loss(loss_type, **kwargs):
    loss_type = loss_type.lower()

    if loss_type in ['binary_cross_entropy', 'binary_crossentropy', 'bin_cross_entropy', 'bin_crossentropy', 'bce']:
        return tf.keras.losses.BinaryCrossentropy(**kwargs)
    if loss_type in ['categorical_cross_entropy', 'categorical_crossentropy', 'cate_cross_entropy',
                     'cate_crossentropy', 'cce']:
        return tf.keras.losses.CategoricalCrossentropy(**kwargs)
    if loss_type in ['categorical_hinge', 'catehinge', 'cate_hinge', 'chinge', 'ch']:
        return tf.keras.losses.CategoricalHinge(**kwargs)
    if loss_type in ['cosine_similarity', 'cossim', 'cos_sim', 'cossimilarity', 'cos_similarity', 'cs']:
        return tf.keras.losses.CosineSimilarity(**kwargs)
    if loss_type in ['hinge']:
        return tf.keras.losses.Hinge(**kwargs)
    if loss_type in ['huber']:
        return tf.keras.losses.Huber(**kwargs)
    if loss_type in ['kullback_leibler_divergence', 'kld']:
        return tf.keras.losses.KLDivergence(**kwargs)
    if loss_type in ['mean_absolute_error', 'mean_abs_error', 'mae']:
        return tf.keras.losses.MeanAbsoluteError(**kwargs)
    if loss_type in ['mean_absolute_percentage_error', 'mean_abs_percent_error', 'mape']:
        return tf.keras.losses.MeanAbsolutePercentageError(**kwargs)
    if loss_type in ['mean_squared_error', 'mean_square_error', 'mse']:
        return tf.keras.losses.MeanSquaredError(**kwargs)
    if loss_type in ['mean_squared_logarithmic_error', 'mean_squared_log_error', 'mean_square_log_error', 'msle']:
        return tf.keras.losses.MeanSquaredLogarithmicError(**kwargs)
    if loss_type in ['sparse_categorical_crossentropy', 'sparse_cate_crossentropy', 'sparse_categorical_cross_entropy',
                     'sparse_cate_crosse_entropy', 'scce']:
        return tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)
    if loss_type in ['bpr', 'bpr_loss']:
        from .extend_losses import BPRLoss
        return BPRLoss(**kwargs)
    if loss_type in ['phinge', 'pair_hinge', 'pair_hinge_loss']:
        from .extend_losses import PairHingeLoss
        return PairHingeLoss(**kwargs)
    return None


def create_metric(metric_type, name_prefix, v1=False, **kwargs):
    metric_type = metric_type.lower()

    if not kwargs or 'name' not in kwargs or not kwargs['name'] or not kwargs['name'].strip():
        kwargs = kwargs or {}
        kwargs['name'] = name_prefix + '_' + metric_type if name_prefix else metric_type
    else:
        kwargs['name'] = name_prefix + '_' + kwargs['name'].strip() if name_prefix else kwargs['name'].strip()

    if metric_type in ['accuracy', 'acc']:
        if v1:
            return tf.compat.v1.metrics.accuracy
        return tf.keras.metrics.Accuracy(**kwargs)
    if metric_type in ['auc']:
        if v1:
            return tf.compat.v1.metrics.auc
        return tf.keras.metrics.AUC(**kwargs)
    if metric_type in ['binary_accuracy', 'binary_acc', 'bin_acc', 'bacc']:
        if v1:
            return None
        return tf.keras.metrics.BinaryAccuracy(**kwargs)
    if metric_type in ['binary_cross_entropy', 'binary_crossentropy', 'bin_cross_entropy', 'bin_crossentropy', 'bce']:
        if v1:
            return None
        return tf.keras.metrics.BinaryCrossentropy(**kwargs)
    if metric_type in ['categorical_accuracy', 'categorical_acc', 'cate_accuracy', 'cate_acc', 'cacc']:
        if v1:
            return None
        return tf.keras.metrics.CategoricalAccuracy(**kwargs)
    if metric_type in ['categorical_cross_entropy', 'categorical_crossentropy', 'cate_cross_entropy',
                       'cate_crossentropy', 'cce']:
        if v1:
            return None
        return tf.keras.metrics.CategoricalCrossentropy(**kwargs)
    if metric_type in ['categorical_hinge', 'catehinge', 'cate_hinge', 'chinge', 'ch']:
        if v1:
            return None
        return tf.keras.metrics.CategoricalHinge(**kwargs)
    if metric_type in ['cosine_similarity', 'cossim', 'cos_sim', 'cossimilarity', 'cos_similarity', 'cs']:
        if v1:
            return None
        return tf.keras.metrics.CosineSimilarity(**kwargs)
    if metric_type in ['kullback_leibler_divergence', 'kld']:
        if v1:
            return None
        return tf.keras.metrics.KLDivergence(**kwargs)
    if metric_type in ['mean_absolute_error', 'mean_abs_error', 'mae']:
        if v1:
            return None
        return tf.keras.metrics.MeanAbsoluteError(**kwargs)
    if metric_type in ['mean_squared_error', 'mean_square_error', 'mse']:
        if v1:
            return None
        return tf.keras.metrics.MeanSquaredError(**kwargs)
    if metric_type in ['mean_absolute_percentage_error', 'mean_abs_percent_error', 'mape']:
        if v1:
            return None
        return tf.keras.metrics.MeanAbsolutePercentageError(**kwargs)
    if metric_type in ['mean_squared_logarithmic_error', 'mean_squared_log_error', 'mean_square_log_error', 'msle']:
        if v1:
            return None
        return tf.keras.metrics.MeanSquaredLogarithmicError(**kwargs)
    if metric_type in ['precision', 'prec']:
        if v1:
            return tf.compat.v1.metrics.precision
        return tf.keras.metrics.Precision(**kwargs)
    if metric_type in ['precision_at_recall', 'prec_at_recall', 'precision@recall', 'prec@recall']:
        if v1:
            return None
        return tf.keras.metrics.PrecisionAtRecall(**kwargs)
    if metric_type in ['recall']:
        if v1:
            return tf.compat.v1.metrics.recall
        return tf.keras.metrics.Recall(**kwargs)
    if metric_type in ['recall_at_precision', 'recall_at_prec', 'recall@precision', 'recall@prec']:
        if v1:
            return None
        return tf.keras.metrics.RecallAtPrecision(**kwargs)
    if metric_type in ['root_mean_squared_error', 'root_mean_square_error', 'rmse']:
        if v1:
            return None
        return tf.keras.metrics.RootMeanSquaredError(**kwargs)
    if metric_type in ['sparse_categorical_accuracy', 'sparse_categorical_acc', 'sparse_cate_accuracy',
                       'sparse_cate_acc', 'scacc']:
        if v1:
            return None
        return tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)
    if metric_type in ['sparse_categorical_crossentropy', 'sparse_cate_crossentropy',
                       'sparse_categorical_cross_entropy', 'sparse_cate_crosse_entropy', 'scce']:
        if v1:
            return None
        return tf.keras.metrics.SparseCategoricalCrossentropy(**kwargs)
    if metric_type in ['sparse_top_k_categorical_accuracy', 'sparse_top_k_categorical_acc',
                       'sparse_top_k_cate_accuracy', 'sparse_top_k_cate_acc', 'sparse_top_k_cacc', 's_top_k_cacc']:
        if v1:
            return None
        return tf.keras.metrics.SparseTopKCategoricalAccuracy(**kwargs)
    if metric_type in ['top_k_categorical_accuracy', 'top_k_categorical_acc', 'top_k_cate_accuracy',
                       'top_k_cate_acc', 'top_k_cacc']:
        if v1:
            return None
        return tf.keras.metrics.TopKCategoricalAccuracy(**kwargs)
    if metric_type in ['f1', 'f1s', 'f1score', 'f1_score']:
        if v1:
            return None
        # from tensorflow_addons.metrics import F1Score
        from .extend_metrics import F1Score
        return F1Score(**kwargs)
    if metric_type in ['gauc', 'grouped_auc']:
        if v1:
            return None
        from .extend_metrics import GroupedAUC
        return GroupedAUC(**kwargs)
    if metric_type in ['po_acc', 'pair_order_acc', 'pair_order_accuracy']:
        if v1:
            return None
        from .extend_metrics import PairOrderAccuracy
        return PairOrderAccuracy(**kwargs)
    return None


def create_activate_func(func_name):
    if not func_name or not func_name.strip():
        return None
    func_name = func_name.strip().lower()
    if func_name == "relu":
        return tf.nn.relu
    elif func_name in ["lrelu", "leaky_relu"]:
        return tf.nn.leaky_relu
    elif func_name == "selu":
        return tf.nn.selu
    elif func_name == "sigmoid":
        return tf.nn.sigmoid
    elif func_name == "tanh":
        return tf.nn.tanh
    elif func_name == 'dice':
        return Dice
    elif func_name == 'prelu':
        return tf.keras.layers.PReLU
    return None


def create_regularizer(l1, l2):
    l1 = l1 if l1 is not None and l1 > 0 else None
    l2 = l2 if l2 is not None and l2 > 0 else None
    if l1 is not None and l2 is not None:
        return tf.keras.regularizers.L1L2(l1, l2)
    elif l1 is not None:
        return tf.keras.regularizers.L1(l1)
    elif l2 is not None:
        return tf.keras.regularizers.L2(l2)
    return None


def squash(inputs):
    vec_squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * inputs
    return vec_squashed


def __get_model_name(path):
    try:
        if not os.path.isdir(path):
            print("model path '{}' not exists, can not get model name".format(path))
            return None
        model = tf.keras.models.load_model(path)
        name = model.name
        print("loaded model from '{}' and get model name '{}'".format(path, name))
        del model
        return name
    except Exception as e:
        import traceback
        print("load trained model from '{}' error: {}:\n{}".format(path, e, traceback.format_exc()))
        return None


def try_get_model_name(model_path):
    from concurrent.futures import ProcessPoolExecutor
    ppe = ProcessPoolExecutor(1)
    f = ppe.submit(__get_model_name, model_path)
    name = f.result()
    ppe.shutdown(wait=True)
    return name


def tf_log_level_from_string(level_str, default_level=None):
    if not level_str:
        return default_level
    level_str = level_str.lower()
    if level_str == 'debug':
        return tf.compat.v1.logging.DEBUG
    if level_str in ['warn', 'warning']:
        return tf.compat.v1.logging.WARN
    if level_str == 'error':
        return tf.compat.v1.logging.ERROR
    if level_str == 'info':
        return tf.compat.v1.logging.INFO

    return default_level


def is_distributed_strategy(dist_strategy: tf.distribute.Strategy):
    return isinstance(dist_strategy, (tf.distribute.experimental.MultiWorkerMirroredStrategy,
                                      tf.distribute.experimental.ParameterServerStrategy,
                                      tf.compat.v1.distribute.experimental.ParameterServerStrategy))
