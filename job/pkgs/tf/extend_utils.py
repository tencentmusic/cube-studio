from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

import tensorflow as tf
from tensorflow.python.eager import context, backprop
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util import nest
from tensorflow.python.keras import backend

from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.engine.compile_utils import *

class PCGrad(tf.keras.optimizers.Optimizer):
    """
    PCGrad: 用于处理多任务下不同任务的梯度优化方向冲突, 导致梯度方向被某一个任务主导的情况
    具体做法为: 计算每一个任务的梯度, 判断任务之间梯度是否冲突(计算余弦相似度, 判断是否为负), 对于有冲突的梯度进行处理, 将其投影到一个方向上
    使用方法为: 
        先实例化一个优化器如Adam, 然后实例化PCGrad时将实例化的Adam作为参数传递给PCGrad
        在计算梯度时, 需要得到不同任务的loss, 然后使用PCGrad的process_gradients方法, 将loss和模型参数传递进去, 返回的是经过PCGrad处理的梯度和模型参数对
        最后要应用梯度时, 使用PCGrad的apply_gradients方法来应用梯度
    """
    def __init__(self, optimizer, name="PCGrad"):
        """optimizer: the optimizer being wrapped
        """
        super(PCGrad, self).__init__(name)
        self.optimizer = optimizer
        self._HAS_AGGREGATE_GRAD = self.optimizer._HAS_AGGREGATE_GRAD
        self.num_tasks = 2

    def set_num_tasks(self, num_tasks):
        self.num_tasks = num_tasks

    def process_gradients(self, loss, var_list, tape):
        # assert type(loss) is list
        if not isinstance(loss, list):
            loss = [loss]

        num_tasks = self.num_tasks if len(loss)==self.num_tasks else len(loss)
        random.shuffle(loss)

        # process gradients
        # Compute per-task gradients.
        grads_task = []
        for l in loss:
            grads = tape.gradient(l, var_list, unconnected_gradients='zero')
            l_grads = []
            for grad in grads:
                if grad is not None:
                    l_grads.append(tf.reshape(grad, [-1]))
            l_grads = tf.concat(l_grads, 0)
            grads_task.append(l_grads)
        grads_task = tf.stack(grads_task, 0)

        '''
        [
            [par1_loss1_grad, par2_loss1_grad, ..., parN_loss1_grad], 
            [par1_loss2_grad, par2_loss2_grad, .... parN_loss2_grad], 
                ...
            [par1_lossM_grad, par2_lossM_grad, .... parN_lossM_grad], 
        ], # flatten parameters and gradients, N is the total number of parameters 
        '''

        # Compute gradient projections.
        def proj_grad(grad_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task*grads_task[k])
                proj_direction = inner_product / \
                    tf.reduce_sum(grads_task[k]*grads_task[k])
                grad_task = grad_task - \
                    tf.minimum(proj_direction, 0.) * grads_task[k]
            return grad_task

        proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)

        # Unpack flattened projected gradients back to their original shapes.
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(var_list):
                grad_shape = var.get_shape()
                flatten_dim = np.prod(
                    [grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                proj_grad = proj_grads_flatten[j][start_idx:start_idx+flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(var_list):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad
                start_idx += flatten_dim
        grads_and_vars = list(zip(proj_grads, var_list))

        return grads_and_vars

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        return self.optimizer.apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

    def _resource_apply_dense(self, grad, var):
        return self.optimizer._resource_apply_dense(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _aggregate_gradients(self, grads_and_vars):
        return self.optimizer._aggregate_gradients(grads_and_vars)

    def _clip_gradients(self, grads):
        return self.optimizer._clip_gradients(grads)

    def get_config(self):
        return self.optimizer.get_config()


class ExtendedLossesContainer(compile_utils.LossesContainer):
    """A container class for losses passed to `Model.compile`."""
    """
    在TensorFlow的LossesContainer基础上增加, 最后能够得到不同loss的列表, 而不仅是所有loss的加和, 的功能; 应用到Model.compile的loss上
    """

    def __init__(self, losses, loss_weights=None, output_names=None):
        super(ExtendedLossesContainer, self).__init__(
            losses, loss_weights, output_names=output_names)

    def __call__(self,
                 y_true,
                 y_pred,
                 sample_weight=None,
                 regularization_losses=None,
                 return_list=None):
        """Computes the overall loss.

        Arguments:
          y_true: An arbitrary structure of Tensors representing the ground truth.
          y_pred: An arbitrary structure of Tensors representing a Model's outputs.
          sample_weight: An arbitrary structure of Tensors representing the
            per-sample loss weights. If one Tensor is passed, it is used for all
            losses. If multiple Tensors are passed, the structure should match
            `y_pred`.
          regularization_losses: Additional losses to be added to the total loss.

        Returns:
          Tuple of `(total_loss, per_output_loss_list)`
        """
        y_true = self._conform_to_outputs(y_pred, y_true)
        sample_weight = self._conform_to_outputs(y_pred, sample_weight)

        if not self._built:
            self.build(y_pred)

        y_pred = nest.flatten(y_pred)
        y_true = nest.flatten(y_true)
        sample_weight = nest.flatten(sample_weight)

        loss_values = []  # Used for gradient calculation.
        loss_metric_values = []  # Used for loss metric calculation.
        batch_dim = None
        zip_args = (y_true, y_pred, sample_weight, self._losses, self._loss_weights,
                    self._per_output_metrics)
        for y_t, y_p, sw, loss_obj, loss_weight, metric_obj in zip(*zip_args):
            if y_t is None or loss_obj is None:  # Ok to have no loss for an output.
                continue

            y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
            sw = apply_mask(y_p, sw, get_mask(y_p))
            loss_value = loss_obj(y_t, y_p, sample_weight=sw)

            loss_metric_value = loss_value
            # Correct for the `Mean` loss metrics counting each replica as a batch.
            if loss_obj.reduction == losses_utils.ReductionV2.SUM:
                loss_metric_value *= ds_context.get_strategy().num_replicas_in_sync

            if batch_dim is None:
                batch_dim = array_ops.shape(y_t)[0]
            if metric_obj is not None:
                metric_obj.update_state(
                    loss_metric_value, sample_weight=batch_dim)

            if loss_weight is not None:
                loss_value *= loss_weight
                loss_metric_value *= loss_weight

            if (loss_obj.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE or
                    loss_obj.reduction == losses_utils.ReductionV2.AUTO):
                loss_value = losses_utils.scale_loss_for_distribution(
                    loss_value)

            loss_values.append(loss_value)
            loss_metric_values.append(loss_metric_value)

        if regularization_losses:
            regularization_losses = losses_utils.cast_losses_to_common_dtype(
                regularization_losses)
            reg_loss = math_ops.add_n(regularization_losses)
            loss_metric_values.append(reg_loss)
            loss_values.append(
                losses_utils.scale_loss_for_distribution(reg_loss))

        if loss_values:
            loss_metric_values = losses_utils.cast_losses_to_common_dtype(
                loss_metric_values)
            total_loss_metric_value = math_ops.add_n(loss_metric_values)
            self._loss_metric.update_state(
                total_loss_metric_value, sample_weight=batch_dim)

            loss_values = losses_utils.cast_losses_to_common_dtype(loss_values)
            if return_list:
                return loss_values
            else:
                total_loss = math_ops.add_n(loss_values)
                return total_loss
        else:
            # Ok for a model to have no compiled loss.
            return array_ops.zeros(shape=())


def is_using_mixed_precision(model):
    if hasattr(tf.keras.mixed_precision, 'global_policy'):
        return tf.keras.mixed_precision.global_policy().compute_dtype == tf.float16
    return tf.keras.mixed_precision.experimental.get_layer_policy(model).loss_scale is not None