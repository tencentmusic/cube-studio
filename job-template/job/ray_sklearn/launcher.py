# -*- coding: utf-8 -*-
import os,sys
base_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_dir)
sys.path.append(os.path.realpath(__file__))

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
'GaussianNB': naive_bayes.MultinomialNB,
'MultinomialNB': naive_bayes.MultinomialNB,
'BernoulliNB': naive_bayes.BernoulliNB,
'Naive Bayes': naive_bayes.MultinomialNB,
'nb': naive_bayes.MultinomialNB,

'KNeighborsClassifier': neighbors.KNeighborsClassifier,
'KNN': neighbors.KNeighborsClassifier,
'knn': neighbors.KNeighborsClassifier,

'LogisticRegression' : linear_model.LogisticRegression,
'LR' : linear_model.LogisticRegression,
'lr' : linear_model.LogisticRegression,

'RandomForestClassifier' : ensemble.RandomForestClassifier,
'Random Forest' : ensemble.RandomForestClassifier,

'DecisionTreeClassifier' : tree.DecisionTreeClassifier,
'Decision Tree' : tree.DecisionTreeClassifier,

'GradientBoostingClassifier' : ensemble.GradientBoostingClassifier,
'gbdt' : ensemble.GradientBoostingClassifier,

'SVC' : svm.SVC,
'SVM' : svm.SVC,
'svc' : svm.SVC,
'svm' : svm.SVC,
}

def model_name_parse(model_name):
    return model_name.replace(' ','').lower()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("sklearn estimator launcher")
    arg_parser.add_argument('--train_csv_file_path', type=str, help="训练集csv，|分割，首行header", default='')
    arg_parser.add_argument('--predict_csv_file_path', type=str, help="预测数据集csv，格式和训练集一致，默认为空，需要predict时填", default='')
    arg_parser.add_argument('--label_name', type=str, help="label的列名，必填", default='')
    arg_parser.add_argument('--model_name', type=str, help="训练用到的模型名称，如lr，必填", default='')
    arg_parser.add_argument('--model_args_dict', type=str, help="模型参数，json格式，默认为空", default='')
    arg_parser.add_argument('--model_file_path', type=str, help="模型文件保存文件名，必填", default='')
    arg_parser.add_argument('--predict_result_path', type=str, help="预测结果保存文件名，默认为空，需要predict时填", default='')

    arg_parser.add_argument('--worker_num', type=str, help="ray worker数量", default=1)

    args = arg_parser.parse_args()
    logging.info("{} args: {}".format(__file__, args))

    support = {model_name_parse(model_name) : SUPPORT_MODELS[model_name] for model_name in SUPPORT_MODELS}
    if model_name_parse(args.model_name) not in support:
        print("support models : " + str(SUPPORT_MODELS.keys))
        raise RuntimeError("your model {} not support".format(args.model_name))
    model = support[model_name_parse(args.model_name)]

    model_args_dict = {}
    if args.model_args_dict:
        model_args_dict = json.loads(args.model_args_dict)

    if not (int(args.worker_num) >=1 and int(args.worker_num)<=10):
        raise RuntimeError("worker_num between 1 and 10")
    worker_num = int(args.worker_num)

    if not args.train_csv_file_path and not args.predict_csv_file_path:
        raise("train_csv_file_path and predict_csv_file_path can not both ba empty")

    if args.train_csv_file_path:
        if not os.path.exists(args.train_csv_file_path):
            raise RuntimeError("train_csv_file_path file not exist")
        train_data = pd.read_csv(args.train_csv_file_path, sep='|', header=0)
        print('train_data.shape : ' + str(train_data.shape))
        if train_data.shape[0] <= 0 or train_data.shape[0] <= 0:
            raise RuntimeError("train data load error")

    if args.predict_csv_file_path:
        if not args.predict_result_path:
            raise RuntimeError("predict_result_path can not be empty")
        predict_result_path = args.predict_result_path

        if not os.path.exists(args.predict_csv_file_path):
            raise RuntimeError("predict_csv_file_path file not exist")
        predict_data = pd.read_csv(args.predict_csv_file_path, sep='|', header=0)
        print('predict_data.shape : ' + str(predict_data.shape))
        if predict_data.shape[0] <= 0 or predict_data.shape[0] <= 0:
            raise RuntimeError("predict data load error")

#    if not os.path.exists(args.model_file_path):
#        raise RuntimeError("must set a exist model_file_path")
    model_file_path = args.model_file_path

    print('train_data.columns : ' + str(train_data.columns))
    if not args.label_name or not args.label_name in train_data.columns:
        raise RuntimeError("label_name illegal")
    label = train_data[args.label_name]
    train_data = train_data.drop(args.label_name, axis=1)

    # 启动ray集群
    init_file = '/app/init.sh'
    from ray_launcher import ray_launcher
    head_service_ip = ray_launcher(worker_num, init_file, 'create')
    print('head_service_ip: ' + head_service_ip)
    if not head_service_ip:
        raise RuntimeError("ray cluster not found")
    os.environ['RAY_ADDRESS'] = head_service_ip

    from ray.util.joblib import register_ray
    register_ray()
    with joblib.parallel_backend('ray'):

        st = time.time()

        if args.train_csv_file_path:
            model = model(**model_args_dict)
            model.fit(train_data, label)
            joblib.dump(model, model_file_path)

        if args.predict_csv_file_path:
            if not args.train_csv_file_path:
                model = joblib.load(model_file_path)
            res = model.predict(predict_data)
            with open(predict_result_path, 'w') as f:
                for line in res:
                    f.write(str(line) + '\n')

    print("succ, cost {}s".format(str(time.time() -st)))

    ray_launcher(worker_num, init_file, 'delete')

