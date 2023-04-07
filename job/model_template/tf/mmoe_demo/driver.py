# coding=utf-8
# @Time     : 2021/02/20 
# @Auther   : kalenchen@tencent.com

import argparse
import time
import random
import os
import json
import logging

from job.model_template.utils import replace_default_config
from job.pkgs.constants import ComponentOutput
from job.pkgs.context import KFJobContext
from job.pkgs.utils import recur_expand_param
from job.pkgs.tf.extend_callbacks import ROCCallback
from job.pkgs.tf.feature_util import ModelInputConfig
from job.pkgs.tf.helperfuncs import create_optimizer

import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
import numpy as np

from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow import tensordot, expand_dims
from tensorflow.keras import layers, Model, initializers, regularizers, activations, constraints, Input

from tensorflow.keras.backend import expand_dims,repeat_elements,sum


SEED = 1

np.random.seed(SEED)

random.seed(SEED)


class MMoE(layers.Layer):

    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(MMoE, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = tf.keras.initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = tf.keras.initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = tf.keras.regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = tf.keras.regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = tf.keras.constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = tf.keras.constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []
        for i in range(self.num_experts):
            self.expert_layers.append(layers.Dense(self.units, activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   bias_initializer=self.expert_bias_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))
        for i in range(self.num_tasks):
            self.gate_layers.append(layers.Dense(self.num_experts, activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 bias_initializer=self.gate_bias_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer, activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))

    def call(self, inputs):

        expert_outputs, gate_outputs, final_outputs = [], [], []
        for expert_layer in self.expert_layers:
            expert_output = expand_dims(expert_layer(inputs), axis=2)
            expert_outputs.append(expert_output)
        expert_outputs = tf.concat(expert_outputs, 2)

        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        for gate_output in gate_outputs:
            expanded_gate_output = expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output, self.units, axis=1)
            final_outputs.append(sum(weighted_expert_output, axis=2))
        return final_outputs


def data_preparation(final_job, train_data_path, test_data_path, model_input_config_file):
    column_names = []
    categorical_columns = []
    label_columns = []
    model_input_desc = ModelInputConfig.parse(model_input_config_file)
    all_inputs = model_input_desc.all_inputs
    for input_desc in all_inputs:
        column_name = input_desc[0]
        dtype = input_desc[1]
        is_label_column = input_desc[7]
        print("column_name: " + str(column_name))
        column_names.append(column_name)
        print(type(input_desc[1]))
        print(type(input_desc[7]))
        if dtype == "string" and not is_label_column:
            print(column_name + " is string column")
            categorical_columns.append(column_name)
        if is_label_column:
            print(column_name + " is label column")
            label_columns.append(column_name)

    train_df = pd.read_csv(train_data_path, sep=',',names=column_names)
    test_df = pd.read_csv(test_data_path, sep=',',names=column_names)
    print("train_df shape:" + str(train_df.shape) + " columns:" + train_df.columns)
    print("test_df shape:" + str(test_df.shape) + " columns:" + test_df.columns)

    train_raw_labels = train_df[label_columns]
    other_raw_labels = test_df[label_columns]
    transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    transformed_test = pd.get_dummies(test_df.drop(label_columns, axis=1), columns=categorical_columns)
    print("transformed_train shape:" + str(transformed_train.shape) + " columns:" + transformed_train.columns)
    print("transformed_test shape:" + str(transformed_test.shape) + " columns:" + transformed_test.columns)
    
    transformed_test_columns_set = set(transformed_test.columns)
    for transformed_train_column in transformed_train.columns:
        if transformed_train_column not in transformed_test_columns_set:
            print(transformed_train_column + " does not show in test set, need to fill missing column in test set")
            transformed_test[transformed_train_column] = 0

    dict_outputs = {}
    dict_train_labels = {}
    dict_other_labels = {}
    for label_column in label_columns:
        train_labels = pd.get_dummies(train_raw_labels[label_column]).rename_axis('ID').values
        test_labels = pd.get_dummies(other_raw_labels[label_column]).rename_axis('ID').values
        dict_outputs[label_column] = train_labels.shape[1]
        dict_train_labels[label_column] = train_labels
        dict_other_labels[label_column] = test_labels

    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    test_data = transformed_test
    test_label = [dict_other_labels[key] for key in sorted(dict_other_labels.keys())]
    train_data = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, test_data, test_label, output_info, label_columns


def train_mmoe(final_job):
    train_data_path = final_job["job_detail"]["train_data"]
    test_data_path = final_job["job_detail"]["test_data"]
    model_input_config_file_path = final_job["job_detail"]["model_input_config_file"]
    train_data, train_label, test_data, test_label, output_info, label_columns \
        = data_preparation(final_job,\
            train_data_path, test_data_path, 
            model_input_config_file_path)
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Test data shape = {}'.format(test_data.shape))
    print(output_info) 

    input_layer = Input(shape=(num_features,))

    num_tasks = len(label_columns)
    expert_units = int(final_job["job_detail"]["model_args"]["expert_units"])
    num_experts = int(final_job["job_detail"]["model_args"]["num_experts"])
    tower_units = int(final_job["job_detail"]["model_args"]["tower_units"])
    tower_activation = final_job["job_detail"]["model_args"]["tower_activation"]

    mmoe_layers = MMoE(units=expert_units, num_experts=num_experts, 
        num_tasks=num_tasks)(input_layer)

    output_layers = []

    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = layers.Dense(units=tower_units, 
            activation=tower_activation,
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = layers.Dense(units=output_info[index][0], 
            name=output_info[index][1], 
            activation='softmax',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    model = Model(inputs=[input_layer], outputs=output_layers)
    model.summary()
    loss = {label_column: 'binary_crossentropy' \
        for label_column in label_columns}

    model.compile(
        loss=loss,
        optimizer=create_optimizer(
            final_job["job_detail"]["train_args"]["optimizer"]["type"], 
            float(final_job["job_detail"]["train_args"]["optimizer"]["args"]["learning_rate"])),
        metrics=['accuracy']
    )
    model.fit(
        x=train_data,
        y=train_label,
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(test_data, test_label),
                test_data=(test_data, test_label)
            )
        ],
        epochs=int(final_job["job_detail"]["train_args"]["epochs"])
    )

    model.save(final_job["job_detail"]["model_path"])
            

def load_default_job_config(data_path, pack_path):
    file_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(file_path, "config_template.json")
    template_py_file = os.path.join(file_path, "template.py")

    with open(cfg_file, 'r') as f:
        def_config = json.load(f)
        def_config = recur_expand_param(def_config, data_path, pack_path)
        def_config['job_detail']['script_name'] = template_py_file
        print("loaded default job config from file '{}', set script_name='{}'"
              .format(cfg_file, template_py_file))
        return def_config


def run_template(job, pack_path, upstream_output_file, export_path, pipeline_id, run_id, creator):
    def_tfjob_cfg = load_default_job_config(export_path, pack_path)
    final_job = replace_default_config(def_tfjob_cfg, job, 
        ignored_keys={})

    print("final job spec: {}".format(final_job))

    loaded_model = None
    model_path = final_job['job_detail'].get('load_model_from', '').strip()

    if not model_path or not os.path.exists(model_path):
        print("begin to launch training job for mmoe template job")
        st = time.perf_counter()
        train_mmoe(final_job)
        print("mmoe template job finished, cost {}s".format(time.perf_counter() - st))


def main(job, pack_path, upstream_output_file, export_path, pipeline_id, run_id, creator):
    ctx = KFJobContext.get_context()
    print("ctx: {}".format(ctx))

    pack_path = pack_path or ctx.pack_path
    export_path = export_path or ctx.export_path
    pipeline_id = pipeline_id or ctx.pipeline_id
    run_id = run_id or ctx.run_id
    creator = creator or ctx.creator
    job = recur_expand_param(job, export_path, pack_path)
    print("expanded user job spec: {}".format(job))

    if job and not job.get("skip"):
        if not os.path.isdir(export_path):
            os.makedirs(os.path.abspath(export_path), exist_ok=True)
            print("{}: created export path '{}'".format(__file__, os.path.abspath(export_path)))

        run_template(job, pack_path, upstream_output_file, export_path, pipeline_id, run_id, creator)
    elif job.get("skip"):
        print("mmoe model template job is skipped")
    else:
        print("empty mmoe model template job")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("mmoe model template")
    arg_parser.add_argument('--job', type=str, required=True, help="模型训练任务描述json")
    arg_parser.add_argument('--pack-path', type=str, help="用户包（包含所有用户文件的目录）的挂载到容器中的路径")
    arg_parser.add_argument('--upstream-output-file', type=str, help="上游输出文件（包含路径）")
    arg_parser.add_argument('--export-path', type=str, help="数据导出目录")
    arg_parser.add_argument('--pipeline-id', type=str, help="pipeline id")
    arg_parser.add_argument('--run-id', type=str, help="运行id，标识每次运行")
    arg_parser.add_argument('--creator', type=str, help="pipeline的创建者")

    args = arg_parser.parse_args()
    print("{} args: {}".format(__file__, args))

    job_spec = json.loads(args.job)
    main(job_spec, args.pack_path, args.upstream_output_file, args.export_path, args.pipeline_id, args.run_id,
         args.creator)