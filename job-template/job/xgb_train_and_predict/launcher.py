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
import re
import subprocess
import copy
import pandas as pd
import numpy as np
import pickle
import sys
import xgboost as xgb

#import seaborn as sns
#import matplotlib
#import matplotlib.pyplot as plt
#import pandas as pd
#import numpy as np
#np.seterr(divide='ignore',invalid='ignore')

#模型效果可视化
#def plotModelResults(prediction, actual):
#    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
#    plt.plot(actual, label="actual", linewidth=2.0)
    
#    lower = - 40
#    upper = prediction + 20

##     plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
##     plt.plot(upper, "r--", alpha=0.5)
 
##     anomalies = np.array([np.NaN] * len(actual))
##     anomalies[prediction < lower] = actual[prediction < lower]
##     anomalies[prediction > upper] = actual[prediction > upper]
##     plt.plot(anomalies, "o", markersize=5, label="Anomalies")

#    plt.legend(loc="best")
#    plt.tight_layout()
#    plt.grid(True);
#    return

##特征权重（仅sklearn线性模型可用）
#def plotCoefficients(model, X_train):
#    coefs = pd.DataFrame(model.coef_, X_train.columns)
#    coefs.columns = ["coef"]
#    coefs["abs"] = coefs.coef.apply(np.abs)
#    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
# 
#    coefs.coef.plot(kind='bar')
#    plt.grid(True, axis='y')
#    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')


if __name__ == "__main__":
    logging.info("raw: {}".format(sys.argv))
    arg_parser = argparse.ArgumentParser("xgb launcher")
#    # XGBClassifier XGBRegressor
    arg_parser.add_argument('--classifier_or_regressor', type=str, help="分类还是回归", default='classifier')
    arg_parser.add_argument('--sep', type=str, help="分隔符", default='')
    arg_parser.add_argument('--params', type=str, help="xgb参数, json格式", default='')
    arg_parser.add_argument('--train_csv_file_path', type=str, help="训练集csv，首行是header，首列是label。为空则不做训练，尝试从model_load_path加载模型。", default='')
    arg_parser.add_argument('--model_load_path', type=str, help="模型加载路径。为空则不加载。", default='')
    arg_parser.add_argument('--predict_csv_file_path', type=str, help="预测数据集csv，格式和训练集一致，顺序保持一致，没有label列。为空则不做predict。", default='')
    arg_parser.add_argument('--predict_result_path', type=str, help="预测结果保存路径，为空则不做predict", default='')
    arg_parser.add_argument('--model_save_path', type=str, help="模型文件保存路径。为空则不保存模型", default='')
    arg_parser.add_argument('--eval_result_path', type=str, help="模型评估报告保存路径。默认为空，想看模型评估报告就填", default='')

    args = arg_parser.parse_args()
    logging.info("{} args: {}".format(__file__, args))

    if args.sep not in ('space', 'TAB', ','):
        raise RuntimeError("args.sep not in ('space', 'TAB', ',')")
    if args.sep == 'space':
        sep = ' '
    if args.sep == 'TAB':
        sep = "\t"
    if args.sep == ',':
        sep = ","
    logging.info('sep: ' + str(sep))

    if args.classifier_or_regressor not in ('classifier', 'regressor'):
        raise RuntimeError("args.classifier_or_regressor not in ('classifier', 'regressor')")
    classifier_or_regressor = args.classifier_or_regressor

    params = json.loads(args.params)

    if args.train_csv_file_path:
        if not os.path.exists(args.train_csv_file_path):
            raise RuntimeError("not os.path.exists(args['train_csv_file_path'])")
        train_data = pd.read_csv(args.train_csv_file_path, sep=sep, header=0, nrows=100)
        logging.info('train_data.shape : ' + str(train_data.shape))
        logging.info('train_data.head(5) : ' + str(train_data.head(5)))
        if train_data.shape[0] <= 0 or train_data.shape[1] <= 0:
            raise RuntimeError("train_data.shape[0] <= 0 or train_data.shape[1] <= 0")
        train_csv_file_path = args.train_csv_file_path
    else:
        train_csv_file_path = None
        train_data = None

    if args.model_load_path:
        if not os.path.exists(args.model_load_path):
            raise RuntimeError("not os.path.exists(args['model_load_path'])")
        model_load_path = args.model_load_path
    else:
        model_load_path = None

    if args.predict_csv_file_path:
        if not os.path.exists(args.predict_csv_file_path):
            raise RuntimeError("not os.path.exists(args['predict_csv_file_path'])")
        predict_data = pd.read_csv(args.predict_csv_file_path, sep=sep, header=0, nrows=100)
        logging.info('predict_data.shape : ' + str(predict_data.shape))
        logging.info('predict_data.head(5) : ' + str(predict_data.head(5)))
        if predict_data.shape[0] <= 0 or predict_data.shape[1] <= 0:
            raise RuntimeError("predict_data.shape[0] <= 0 or predict_data.shape[1] <= 0")
        if predict_data.shape[1] != train_data.shape[1] -1:
            raise RuntimeError("predict_data.shape[1] != train_data.shape[1] -1")
        predict_csv_file_path = args.predict_csv_file_path
    else:
        predict_data = None
        predict_csv_file_path = None

    if args.predict_result_path:
        if not os.path.exists(os.path.split(args.predict_result_path)[0]):
            raise RuntimeError("not os.path.exists(os.path.split(args['predict_result_path'])[0])")
        predict_result_path = args.predict_result_path
    else:
        predict_result_path = None

    if args.model_save_path:
        if not os.path.exists(os.path.split(args.model_save_path)[0]):
            raise RuntimeError("not os.path.exists(os.path.split(args.model_save_path)[0])")
        model_save_path = args.model_save_path
    else:
        model_save_path = None

    if args.eval_result_path:
        if not os.path.exists(os.path.split(args.eval_result_path)[0]):
            raise RuntimeError("not os.path.exists(os.path.split(args['eval_result_path'])[0])")
        eval_result_path = args.eval_result_path
    else:
        eval_result_path = None

    if classifier_or_regressor == 'classifier':
        estimator = xgb.XGBRegressor
    else:
        estimator = xgb.XGBClassifier

    if model_load_path:
        logging.info("开始加载模型 model_load_path: " + str(model_load_path))
        mod = pickle.load(open(model_load_path, "rb"))
        logging.info("加载完了")

    elif not train_data.empty:
        logging.info("开始训练模型 train_data: " + str(train_data))
        train_data = pd.read_csv(train_csv_file_path, sep=sep, header=0)
        train_data = train_data.infer_objects()
        col = train_data.columns
        label_name = col[0]
        logging.info("label_name: " + str(label_name))
        y = train_data[label_name]
        X = train_data.drop(label_name, axis=1)
        mod = estimator(**params)
        mod = mod.fit(X, y)
        logging.info("训练完了")
    else:
        logging.error("没加载模型，又没训练模型，参数有问题。")

    if model_save_path:
        logging.info("开始保存模型。")
        pickle.dump(mod, open(model_save_path, "wb"))
        logging.info("保存模型完成。")

    if predict_csv_file_path and predict_result_path:
        logging.info("开始预测 predict_data: " + str(predict_data))
        predict_data = pd.read_csv(predict_csv_file_path, sep=sep, header=0)
        predict_data = predict_data.infer_objects()
        res = mod.predict(predict_data)
        logging.info(res)
        res_f = open(predict_result_path, "w")
        for x in res:
            res_f.write(str(x) + '\n')
        logging.info("预测完了")

    if eval_result_path:
        if train_data.empty:
            logging.warning("你又没训练，不帮你评估了")
        else:
            logging.info("开始评估模型")

#            plt.figure(figsize=(10, 8))
#            plt.subplot(211)
#            plotModelResults(prediction, actual)
#
#            plt.subplot(212)
#            plotCoefficients(lr, X_train=f.X_train)
#            savefig()
            logging.info("评估模型完了")








