'''
Author: your name
Date: 2021-06-10 11:04:39
LastEditTime: 2021-06-17 15:57:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /job-template/job/pkgs/tf/custom/custom_layers.py
'''
# coding=utf-8
import tensorflow as tf

class PersonalRadioExpertV1(tf.keras.layers.Layer):
    def __init__(self, layer_id, task_name, expert_id, name='prev1_layer', trainable=True, **kwargs):
        super(PersonalRadioExpertV1, self).__init__(name=name, trainable=trainable, **kwargs)
        self.layer_id = layer_id # ple层的id
        self.task_name = task_name # 多任务的任务名称
        self.expert_id = expert_id # 专家id
        self.d1 = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu', name='L%d_T%s_E%d_dense_relu_1' % (self.layer_id, self.task_name, self.expert_id))
        self.d2 = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu', name='L%d_T%s_E%d_dense_relu_2' % (self.layer_id, self.task_name, self.expert_id))
        self.d3 = tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu', name='L%d_T%s_E%d_dense_relu_3' % (self.layer_id, self.task_name, self.expert_id))
        self.res = tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.01), name='L%d_T%s_E%d_residual_dense4' % (self.layer_id, self.task_name, self.expert_id))

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        y = inputs
        y = tf.concat([y, self.d1(y), self.d2(y), self.d3(y)], axis=1)
        y = self.res(y)
        return y

    def get_config(self):
        config = super(PersonalRadioExpertV1, self).get_config()
        config.update({
            'layer_id': self.layer_id,
            'task_name': self.task_name,
            'expert_id': self.expert_id,
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PersonalRadioInputDropoutV1(tf.keras.layers.Layer):
    def __init__(self, name='pridv1_layer', trainable=True, **kwargs):
        super(PersonalRadioInputDropoutV1, self).__init__(name=name, trainable=trainable, **kwargs)
        self.dr = tf.keras.layers.Dropout(0.5, name='dropout_r')
        
    @tf.function
    def call(self, inputs, training=None, **kwargs):
        y = inputs
        y = self.dr(inputs, training=training)
        return y

    def get_config(self):
        config = super(PersonalRadioInputDropoutV1, self).get_config()
        config.update({
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PersonalRadioTowerV1(tf.keras.layers.Layer):
    def __init__(self, task_name, name='prtv1', trainable=True, **kwargs):
        super(PersonalRadioTowerV1, self).__init__(name=name, trainable=trainable, **kwargs)
        self.task_name = task_name
        self.seq1 = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu', name='T%s_tower_dense' % (task_name))
        self.seq2 = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='sigmoid', name='T%s_sigmoid_out' % (task_name))
        
    @tf.function
    def call(self, inputs, training=None, **kwargs):
        y = inputs
        y = self.seq1(y)
        y = self.seq2(y)
        return y

    def get_config(self):
        config = super(PersonalRadioTowerV1, self).get_config()
        config.update({
            'task_name': self.task_name,
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
