# -*- coding: utf-8 -*-
import os,sys
base_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_dir)
sys.path.append(os.path.realpath(__file__))
import pysnooper
from common import logging, nonBlockRead, HiddenPrints

import argparse
import datetime
import json
import time
import uuid
#import pysnooper
import re
import subprocess
#import psutil
import copy
import pandas as pd
import numpy as np

from sklearn import naive_bayes, neighbors, linear_model, ensemble, tree, svm
import joblib

SUPPORT_MODELS = {
    'GaussianNB': naive_bayes.GaussianNB,
    'MultinomialNB': naive_bayes.MultinomialNB,
    'BernoulliNB': naive_bayes.BernoulliNB,
    'CategoricalNB':naive_bayes.CategoricalNB,
    'ComplementNB': naive_bayes.ComplementNB,
    'KNeighborsClassifier': neighbors.KNeighborsClassifier,
    'LogisticRegression' : linear_model.LogisticRegression,
    'RandomForestClassifier' : ensemble.RandomForestClassifier,
    'DecisionTreeClassifier' : tree.DecisionTreeClassifier,
    'GradientBoostingClassifier' : ensemble.GradientBoostingClassifier,
    'SVC' : svm.SVC,
    'SVR' : svm.SVR
}

def model_name_parse(model_name):
    return model_name.replace(' ','').lower()

# @pysnooper.snoop()
def start(args):
    # 处理label列和特征列
    label_columns = args.label_columns.split(',') if args.label_columns else []
    feature_columns = args.feature_columns.split(',') if args.feature_columns else []

    # 获取模型
    support = {model_name_parse(model_name): SUPPORT_MODELS[model_name] for model_name in SUPPORT_MODELS}
    if model_name_parse(args.model_name) not in support:
        print("support models : " + str(SUPPORT_MODELS.keys))
        raise RuntimeError("your model {} not support".format(args.model_name))
    model = support[model_name_parse(args.model_name)]

    # 获取模型参数
    model_args_dict = {}
    if args.model_args_dict:
        model_args_dict = json.loads(args.model_args_dict)

    # 并发数限制
    if not (int(args.worker_num) >= 1 and int(args.worker_num) <= 10):
        raise RuntimeError("worker_num between 1 and 10")
    worker_num = int(args.worker_num)

    # 至少做训练和推理中的一件事
    if not args.train_csv_file_path and not args.predict_csv_file_path:
        raise ("train_csv_file_path and predict_csv_file_path can not both ba empty")

    # 模型地址必填
    model_file_path = args.model_file_path
    os.makedirs(os.path.dirname(model_file_path),exist_ok=True)

    # 加载训练文件
    train_data = None
    if args.train_csv_file_path:
        if not os.path.exists(args.train_csv_file_path):
            raise RuntimeError("train_csv_file_path file not exist")
        train_data = pd.read_csv(args.train_csv_file_path, sep=',', header=0)
        print('train_data.shape : ' + str(train_data.shape))
        if train_data.shape[0] <= 0 or train_data.shape[1] <= 0:
            raise RuntimeError("train data load error")

    # 加载推理文件
    predict_data = None
    predict_result_path = args.predict_result_path
    if args.predict_csv_file_path:
        if not os.path.exists(args.predict_csv_file_path):
            raise RuntimeError("predict_csv_file_path file not exist")
        if not args.predict_result_path:
            raise RuntimeError("predict_result_path can not be empty")
        predict_data = pd.read_csv(args.predict_csv_file_path, sep=',', header=0)
        print('predict_data.shape : ' + str(predict_data.shape))
        if predict_data.shape[0] <= 0 or predict_data.shape[1] <= 0:
            raise RuntimeError("predict data load error")

    # 如果做训练就校验下label列和特征列
    label_data = None
    if args.train_csv_file_path:
        print('train_data.columns : ' + str(train_data.columns))
        # 标签列必须要有，且在数据列中
        if not label_columns or not [item in label_columns for item in train_data.columns.tolist()]:
            raise RuntimeError("label_columns illegal")
        # 特征列可以没有，但是如果填了就必须在数据列中
        if feature_columns and not [item in feature_columns for item in train_data.columns.tolist()]:
            raise RuntimeError("feature_columns illegal")
        # 获取label数据和训练数据
        label_data = train_data[label_columns]
        if not args.feature_columns:
            train_data = train_data.drop(label_columns, axis=1)
        else:
            train_data = train_data[feature_columns]

    # 启动ray集群
    init_file = '/app/init.sh'
    if args.init_file:
        init_file=args.init_file
    from job.pkgs.k8s.py_ray import ray_launcher
    head_service_ip = ray_launcher(worker_num, init_file, 'create')
    print('head_service_ip: ' + head_service_ip)
    if not head_service_ip:
        raise RuntimeError("ray cluster not found")
    os.environ['RAY_ADDRESS'] = head_service_ip

    from ray.util.joblib import register_ray

    # 注册 Ray 作为 Joblib 的后端
    register_ray()
    with joblib.parallel_backend('ray'):

        st = time.time()

        if args.train_csv_file_path:
            model = model(**model_args_dict)
            model.fit(train_data, label_data)
            joblib.dump(model, model_file_path)

        if args.predict_csv_file_path:
            model = joblib.load(model_file_path)
            res = model.predict(predict_data)
            with open(predict_result_path, 'w') as f:
                for line in res:
                    f.write(str(line) + '\n')

    print("succ, cost {}s".format(str(time.time() - st)))

    ray_launcher(worker_num, init_file, 'delete')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("sklearn estimator launcher")
    arg_parser.add_argument('--train_csv_file_path', type=str, help="训练集csv，|分割，首行header", default='')
    arg_parser.add_argument('--predict_csv_file_path', type=str, help="预测数据集csv，格式和训练集一致，默认为空，需要predict时填", default='')
    arg_parser.add_argument('--feature_columns', type=str, help="特征的列名，必填", default='')
    arg_parser.add_argument('--label_columns', type=str, help="label的列名，必填", default='')
    arg_parser.add_argument('--model_name', type=str, help="训练用到的模型名称，如lr，必填", default='')
    arg_parser.add_argument('--model_args_dict', type=str, help="模型参数，json格式，默认为空", default='')
    # arg_parser.add_argument('--model_param_space', type=str, help="模型超参数空间，json格式，默认为空", default='')
    arg_parser.add_argument('--model_file_path', type=str, help="模型文件保存文件名，必填", default='')
    arg_parser.add_argument('--predict_result_path', type=str, help="预测结果保存文件名，默认为空，需要predict时填", default='')
    arg_parser.add_argument('--init_file', type=str, help="初始化文件地址", default='')
    arg_parser.add_argument('--worker_num', type=str, help="ray worker数量", default=1)

    args = arg_parser.parse_args()
    logging.info("{} args: {}".format(__file__, args))
    start(args)
