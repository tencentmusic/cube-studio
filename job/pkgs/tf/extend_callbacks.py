# coding=utf-8
# @Time     : 2020/12/9 14:49
# @Auther   : lionpeng@tencent.com

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.python.distribute import (collective_all_reduce_strategy,
                                          distribute_lib, mirrored_strategy)

try:
    from tensorflow.python.keras.distribute import worker_training_state
except:
    from . import worker_training_state

import time

import nni


class NNISearchPushCallBack(tf.keras.callbacks.Callback):
    def __init__(self, nni_exp_id, nni_trial_id, nni_record, every_batch=None, nni_metric=None):
        super(NNISearchPushCallBack, self).__init__()

        self.nni_exp_id = nni_exp_id
        self.nni_trial_id = nni_trial_id
        self.nni_record = nni_record
        self.every_batch = every_batch
        self.nni_metric = nni_metric

    def on_epoch_end(self, epoch, logs=None):
        # 在每个epoch结束后，对于模型在validation上的结果进行NNI中间结果保存
        metrics = logs
        nni_metric = 'val_'+self.nni_metric if self.nni_metric is not None else self.nni_metric
        if nni_metric is None or nni_metric not in metrics:
            report_metric = list(metrics.values())[-1]
        else:
            report_metric = metrics.get(nni_metric)
        with open(self.nni_record,'a+') as f:
            f.write('intermediate_{}_\n'.format(report_metric))
        print('nni epoch end intermediate', epoch+1, report_metric, logs)

    def on_train_end(self, logs=None):
        # 在每次训练结束之后，对于模型在validation上的结果进行NNI最终结果保存
        metrics = logs
        nni_metric = 'val_'+self.nni_metric if self.nni_metric is not None else self.nni_metric
        if nni_metric is None or nni_metric not in metrics:
            report_metric = list(metrics.values())[-1]
        else:
            report_metric = metrics.get(nni_metric)
        with open(self.nni_record,'a+') as f:
            f.write('final_{}_\n'.format(report_metric))
        print('nni train end final', report_metric, logs)

    def on_predict_end(self, logs=None):
        # 如果进行了evaluate，则对模型在validation上的结果进行NNI最终结果保存
        metrics = logs
        if self.nni_metric is None or self.nni_metric not in metrics:
            report_metric = list(metrics.values())[-1]
        else:
            report_metric = metrics.get(self.nni_metric)
        with open(self.nni_record,'a+') as f:
            f.write('final_{}_\n'.format(report_metric))
        print('nni predict end final', report_metric, logs)


class BackupAndStockCallBack(tf.keras.callbacks.Callback):
    def __init__(self, backup_dir):
        super(BackupAndStockCallBack, self).__init__()
        self.backup_dir = backup_dir
        self._supports_tf_logs = True
        self._supported_strategies = (
            distribute_lib._DefaultDistributionStrategy,
            mirrored_strategy.MirroredStrategy,
            collective_all_reduce_strategy.CollectiveAllReduceStrategy)
        # Only the chief worker writes model checkpoints, but all workers
        # restore checkpoint at on_train_begin().
        self._chief_worker_only = False

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        # TrainingState is used to manage the training state needed for
        # failure-recovery of a worker in training.
        # pylint: disable=protected-access

        if not isinstance(self.model.distribute_strategy,
                          self._supported_strategies):
            raise NotImplementedError(
                'Currently only support empty strategy, MirroredStrategy and '
                'MultiWorkerMirroredStrategy.')
        self.model._training_state = (
            worker_training_state.WorkerTrainingState(self.model, self.backup_dir))
        self._training_state = self.model._training_state
        self._training_state.restore()

    def on_train_end(self, logs=None):
        # pylint: disable=protected-access
        # On exit of training, delete the training state backup file that was saved
        # for the purpose of worker recovery.

        # Clean up the training state.
        del self._training_state
        del self.model._training_state

    def on_epoch_end(self, epoch, logs=None):
        # Back up the model and current epoch for possible future recovery.
        self._training_state.back_up(epoch)


class TrainSpeedLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, every_batches=100, with_metrics=False):
        super(TrainSpeedLoggerCallback, self).__init__()
        self.every_batches = max(int(every_batches), 1)
        self.total_batches = 0
        self.total_time = 0
        self.start_time = None
        self.with_metrics = with_metrics

    def on_train_batch_begin(self, batch, logs=None):
        if self.start_time is None:
            self.start_time = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        self.total_batches += 1
        if self.total_batches % self.every_batches == 0:
            elapsed = time.perf_counter() - self.start_time
            self.total_time += elapsed
            rt_speed = self.every_batches/elapsed
            avg_speed = self.total_batches/self.total_time
            tf.print("step {}(batch #{}): total cost time {}s, {} batchs cost {}s, rt step/sec: {}, avg step/sec: {}"
                     .format(self.total_batches, batch, self.total_time, self.every_batches, elapsed, rt_speed,
                             avg_speed))
            if self.with_metrics:
                tf.print("step {}(batch #{}): {}".format(self.total_batches, batch, logs))
            self.start_time = None


class ROCCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            print(
                'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                    output_name, round(train_roc_auc, 4),
                    output_name, round(validation_roc_auc, 4),
                    output_name, round(test_roc_auc, 4)
                )
            )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
