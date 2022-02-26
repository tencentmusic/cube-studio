
from abc import abstractmethod, ABC

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.engine.compile_utils import MetricsContainer, match_dtype_and_rank, get_mask, apply_mask
from tensorflow.python.util import nest
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.util import dispatch
import time


class ExtendedMetric(tf.keras.metrics.Metric):
    def __init__(self, name=None, dtype=None, **kwargs):
        super(ExtendedMetric, self).__init__(name, dtype, **kwargs)

    @abstractmethod
    def update_state(self, y_true, y_pred, sample_weight, x):
        raise NotImplementedError("must be implemented in subclasses")


class CumulationMetric(ExtendedMetric):
    def __init__(self, name=None, dtype=None, **kwargs):
        super(CumulationMetric, self).__init__(name, dtype, **kwargs)

    @abstractmethod
    def final_result(self):
        raise NotImplementedError("must be implemented in subclasses")


@dispatch.add_dispatch_support
def sparse_top_k_interests_m_categorical_accuracy(y_true, y_pred, k=5, m=None):
    """
    在tensorflow的sparse_top_k_categorical_accuracy函数基础上, 增加了对于多个兴趣下命中率的判断
    如果有多个兴趣, 那么会对每个兴趣得到的候选列表分别进行in_top_k计算, 只要有一个兴趣中存在命中的则算作一次命中
    """
    y_pred_rank = tf.convert_to_tensor(y_pred).shape.ndims
    y_true_rank = tf.convert_to_tensor(y_true).shape.ndims
    
    interests = m or y_pred.shape[-2]
    
    if (y_true_rank is not None) and (y_pred_rank is not None):
        if y_pred_rank > 2:
            y_pred = tf.reshape(y_pred, [-1, interests, y_pred.shape[-1]])
        elif y_pred_rank == 2:
            y_pred = tf.reshape(y_pred, [-1, 1, y_pred.shape[-1]])
            interests = 1
        if y_true_rank > 1:
            y_true = tf.reshape(y_true, [-1])
    
    interests_hit = []
    for i in range(interests):
        interests_hit.append(tf.cast(tf.nn.in_top_k(tf.cast(y_true, tf.int32), y_pred[:,i,:], k), tf.float32))
    interests_hit = tf.reduce_sum(interests_hit, 0)
    interests_hit = tf.where(interests_hit>0, 1., interests_hit)
    
    return interests_hit


class SparseTopKInterestsMCategoricalAccuracy(MeanMetricWrapper):
    """
    在tensorflow的SparseTopKCategoricalAccuracy基础上, 增加了对于多个兴趣下命中率的判断
    """
    def __init__(self, k=5, m=None, name='sparse_top_k_interests_m_categorical_accuracy', dtype=None):
        super(SparseTopKInterestsMCategoricalAccuracy, self).__init__(
            sparse_top_k_interests_m_categorical_accuracy, name, dtype=dtype, k=k, m=m)


class GroupedAUC(CumulationMetric):
    def __init__(self, user_id_index, from_logits=False, name="gauc", dtype=None, sample_size=1024):

        super(GroupedAUC, self).__init__(name=name, dtype=dtype)
        self.user_id_index = user_id_index
        self.from_logits = from_logits
        self.sample_size = sample_size
        self.cumu_y_true = None
        self.cumu_y_pred = None
        self.cumu_sample_weight = None
        self.cumu_uids = None

    def update_state(self, y_true, y_pred, sample_weight, x):
        def __cumu(y_t, y_p, uids, sw):
            self.cumu_y_true = tf.concat([self.cumu_y_true, tf.squeeze(y_t)], axis=0)
            self.cumu_y_pred = tf.concat([self.cumu_y_pred, tf.squeeze(y_p)], axis=0)
            self.cumu_sample_weight = None if sw is None else \
                tf.concat([self.cumu_sample_weight, tf.squeeze(sw)], axis=0)
            self.cumu_uids = tf.concat([self.cumu_uids, tf.squeeze(uids)], axis=0)

        def __init(y_t, y_p, uids, sw):
            self.cumu_y_true = tf.squeeze(y_t)
            self.cumu_y_pred = tf.squeeze(y_p)
            self.cumu_sample_weight = None if sw is None else tf.squeeze(sw)
            self.cumu_uids = tf.squeeze(uids)

        def __update_with_weights(y_t, y_p, uids, sw):
            if any(t is None for t in [self.cumu_y_true, self.cumu_y_pred, self.cumu_uids]):
                __init(y_t, y_p, uids, sw)
            else:
                __cumu(y_t, y_p, uids, sw)

        def __update_without_weights(y_t, y_p, uids):
            if any(t is None for t in [self.cumu_y_true, self.cumu_y_pred, self.cumu_uids]):
                __init(y_t, y_p, uids, None)
            else:
                __cumu(y_t, y_p, uids, None)

        if sample_weight is None:
            tf.py_function(__update_without_weights, inp=[y_true, y_pred, x[self.user_id_index]], Tout=[])
        else:
            tf.py_function(__update_with_weights, inp=[y_true, y_pred, x[self.user_id_index], sample_weight], Tout=[])

    def _calc_gauc(self, y_t, y_p, uids, sw=None, sample=True):
        y_t = y_t.numpy()
        y_p = y_p.numpy()
        uids = uids.numpy()
        if sw is not None:
            sw = sw.numpy()

        if sample:
            sample_size = min(y_t.shape[0], self.sample_size)
            sample_indices = np.random.choice(range(y_t.shape[0]), size=sample_size, replace=False)
            y_t = y_t[sample_indices]
            y_p = y_p[sample_indices]
            uids = uids[sample_indices]
            if sw is not None:
                sw = sw[sample_indices]

        if not y_t.shape[0]:
            return 0.

        unique_uids, uid_idxs = tf.unique(uids)
        unique_uids = unique_uids.numpy()
        uid_idxs = uid_idxs.numpy()
        if self.from_logits:
            y_p = 1.0/(1+np.exp(-y_p))

        if np.any(y_t < 0):
            tf.print("y_t contains neg values: {}\n".format(y_t))
            raise RuntimeError("y_t contains neg values: {}".format(y_t))

        if np.any(y_p < 0):
            tf.print("y_p contains neg values: {}\n".format(y_p))
            raise RuntimeError("yp contains neg values: {}".format(y_p))

        valid_num = 0
        overall_auc = 0.
        skip_count = 0
        user_count = 0
        for i, uid in enumerate(unique_uids):
            user_y_true = y_t[uid_idxs == i]
            user_y_pred = y_p[uid_idxs == i]
            user_sample_weights = sw[uid_idxs == i] if sw is not None else None

            if user_y_true.std() == 0:
                skip_count += 1
                continue
            user_auc = roc_auc_score(user_y_true, user_y_pred, sample_weight=user_sample_weights)
            overall_auc += user_auc * user_y_true.shape[0]
            valid_num += user_y_true.shape[0]
            user_count += 1
        if valid_num == 0:
            return 0.
        return overall_auc / valid_num

    def result(self):
        def __calc():
            if any(t is None for t in [self.cumu_y_true, self.cumu_y_pred, self.cumu_uids]):
                return 0.
            return self._calc_gauc(self.cumu_y_true, self.cumu_y_pred, self.cumu_uids, self.cumu_sample_weight, True)

        return tf.py_function(__calc, inp=[], Tout=tf.float32)

    def final_result(self):
        def __calc():
            if any(t is None for t in [self.cumu_y_true, self.cumu_y_pred, self.cumu_uids]):
                return 0.
            return self._calc_gauc(self.cumu_y_true, self.cumu_y_pred, self.cumu_uids, self.cumu_sample_weight, False)

        print("{}: start calculating final gauc, cumu_y_true.shape={}, cumu_y_pred.shape={}, cumu_uids.shape={} ..."
              .format(self.name, self.cumu_y_true.shape, self.cumu_y_pred.shape, self.cumu_uids.shape), flush=True)
        st = time.perf_counter()
        gauc = tf.py_function(__calc, inp=[], Tout=tf.float32)
        print("{}: final gauc calculation complete, cumu_y_true.shape={}, cumu_y_pred.shape={}, cumu_uids.shape={},"
              " cost {}s: {}".format(self.name, self.cumu_y_true.shape, self.cumu_y_pred.shape, self.cumu_uids.shape,
                                     time.perf_counter()-st, gauc), flush=True)
        return gauc

    def reset_states(self):
        self.cumu_y_true = None
        self.cumu_y_pred = None
        self.cumu_sample_weight = None
        self.cumu_uids = None

    def get_config(self):
        config = super(GroupedAUC, self).get_config()
        config.update({
            "user_id_index": self.user_id_index,
            "from_logits": self.from_logits,
            "sample_size": self.sample_size
        })
        return config


class F1Score(tf.keras.metrics.Metric, ABC):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):
        super(F1Score, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=tf.keras.initializers.Zeros)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=tf.keras.initializers.Zeros)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=tf.keras.initializers.Zeros)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        result = tf.math.divide_no_nan(2 * self.true_positives,
                                       self.false_negatives + self.false_positives + 2 * self.true_positives)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        tf.keras.backend.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = super(F1Score, self).get_config()
        config.update({
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        })
        return config


def pair_order_acc(y_true, y_pred, **kwargs):
    # [batch, 1]
    acc = tf.cast(y_pred > 0, tf.float32)
    return acc


class PairOrderAccuracy(MeanMetricWrapper):
    def __init__(self, name="pair_order_acc", dtype=None):
        super(PairOrderAccuracy, self).__init__(pair_order_acc, name=name, dtype=dtype)

    def get_config(self):
        config = super(PairOrderAccuracy, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ExtendedMetricsContainer(MetricsContainer):
    def update_state(self, y_true, y_pred, sample_weight=None, x=None):
        y_true = self._conform_to_outputs(y_pred, y_true)
        sample_weight = self._conform_to_outputs(y_pred, sample_weight)

        if not self._built:
            self.build(y_pred, y_true)

        y_pred = nest.flatten(y_pred)
        y_true = nest.flatten(y_true) if y_true is not None else []
        sample_weight = nest.flatten(sample_weight)

        zip_args = (y_true, y_pred, sample_weight, self._metrics,
                    self._weighted_metrics)
        for y_t, y_p, sw, metric_objs, weighted_metric_objs in zip(*zip_args):
            # Ok to have no metrics for an output.
            if (y_t is None or (all(m is None for m in metric_objs) and
                                all(wm is None for wm in weighted_metric_objs))):
                continue

            y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
            mask = get_mask(y_p)
            sw = apply_mask(y_p, sw, mask)

            for metric_obj in metric_objs:
                if metric_obj is None:
                    continue
                if isinstance(metric_obj, ExtendedMetric):
                    metric_obj.update_state(y_t, y_p, sample_weight=sw, x=x)
                else:
                    metric_obj.update_state(y_t, y_p, sample_weight=sw)

            for weighted_metric_obj in weighted_metric_objs:
                if weighted_metric_obj is None:
                    continue
                if isinstance(weighted_metric_obj, ExtendedMetric):
                    weighted_metric_obj.update_state(y_t, y_p, sample_weight=sw, x=x)
                else:
                    weighted_metric_obj.update_state(y_t, y_p, sample_weight=sw)
