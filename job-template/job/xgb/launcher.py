import shutil
# 加载相关的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

import pysnooper
import os, sys
import argparse
import pandas, json

import os
import pickle
import xgboost as xgb


# 加载IMDb数据集
# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz


def draw_roc_curve(y_true, y_pred, path):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.title('ROC curve')
    plt.plot(fpr, tpr, '#9400D3', label=u'AUC = %0.3f' % roc_auc)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.savefig(path)
    plt.close()


@pysnooper.snoop()
def inference(save_model_dir, inference_dataset, feature_columns):
    save_model_path = os.path.join(save_model_dir, 'xgb_model.pkl')

    with open(save_model_path, 'rb') as file:
        model = pickle.load(file)
        df = pandas.read_csv(inference_dataset)
        feature_columns = [x.strip() for x in feature_columns.split(',') if x.strip()]
        X = df[feature_columns]  # 目标变量

        dinf = xgb.DMatrix(X)

        y_pred = model.predict(dinf)
        print('预测值')
        print(y_pred)
        y_df = pd.DataFrame(y_pred, columns=['y'])

        result = pd.concat([X, y_df], axis=1)

        result.to_csv(os.path.join(save_model_dir, 'inference_result.csv'), index=False, header=True)


@pysnooper.snoop()
def val(save_model_dir, val_dataset, label_columns, feature_columns):
    save_model_path = os.path.join(save_model_dir, 'xgb_model.pkl')
    save_val_path = os.path.join(save_model_dir, 'val_result.json')

    with open(save_model_path, 'rb') as file:
        model = pickle.load(file)
        df = pandas.read_csv(val_dataset)
        label_columns = [x.strip() for x in label_columns.split(',') if x.strip()]
        feature_columns = [x.strip() for x in feature_columns.split(',') if x.strip()]

        X_val = df[feature_columns]  #
        y_val = df[label_columns]  # 目标变量

        dval = xgb.DMatrix(X_val, y_val)

        y_pred = model.predict(dval)

        # 评估模型
        # accuracy = accuracy_score(y_val, y_pred)
        print('预测值')
        print(y_pred)

        auc = roc_auc_score(y_val, y_pred)

        draw_roc_curve(y_val, y_pred, os.path.join(save_model_dir, 'val_roc.png'))

        # print('Accuracy:', accuracy)
        print('Auc:', auc)

        train_test = json.load(open(save_val_path, "r")) if os.path.exists(save_val_path) else {}
        print(train_test)
        train_test.update({"val_auc": auc})
        json.dump(train_test, open(save_val_path, "w"))

        metrics = [
            {
                "metric_type": "image",
                "describe":"验证集ROC曲线",
                "image": os.path.join(save_model_dir, 'val_roc.png')
            }
        ]
        json.dump(metrics, open('/metric.json', mode='w'))


# 训练
@pysnooper.snoop()
def train(save_model_dir, train_dataset, label_columns, feature_columns, model_params):
    # 读取数据
    # if '.csv' in dataset_path:
    df = pandas.read_csv(train_dataset)
    label_columns = [x.strip() for x in label_columns.split(',') if x.strip()]
    feature_columns = [x.strip() for x in feature_columns.split(',') if x.strip()]

    # 处理数据
    # X = df.drop(label_columns, axis=1)  # 特征变量
    X = df[feature_columns]  # 目标变量
    y = df[label_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分训练集和测试集
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    evallist = [(dtrain, 'train'), (dtest, 'test')]

    print(model_params, type(model_params))
    # 训练xgb模型
    model = xgb.train(model_params, dtrain, evals=evallist)

    # 计算auc
    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)
    # 计算auc
    auc_train = roc_auc_score(y_train, y_train_pred)
    auc_test = roc_auc_score(y_test, y_test_pred)
    # print(auc_train,auc_test)
    # print(y_train_pred)

    # 计算acc
    # y_train_pred = model.predict(dtrain)
    # y_test_pred = model.predict(dtest)
    # train_acc = accuracy_score(y_train, y_train_pred)
    # test_acc = accuracy_score(y_test, y_test_pred)

    os.makedirs(save_model_dir, exist_ok=True)

    draw_roc_curve(y_train, y_train_pred, os.path.join(save_model_dir, 'train_roc.png'))
    draw_roc_curve(y_test, y_test_pred, os.path.join(save_model_dir, 'test_roc.png'))

    save_model_path = os.path.join(save_model_dir, 'xgb_model.pkl')
    with open(save_model_path, 'wb') as file:
        pickle.dump(model, file)

    # file = open(os.path.join(save_model_dir,'val_result.json'), "w")
    file = open(os.path.join(save_model_dir, 'val_result' + json.dumps(model_params) + '.json'), "w")
    file.write(json.dumps({"train_auc": auc_train, "test_auc": auc_test}))
    file.close()

    metrics = [
        {
            "metric_type": "image",
            "describe":"训练集ROC曲线",
            "image": os.path.join(save_model_dir, 'train_roc.png')
        },
        {
            "metric_type": "image",
            "describe":"测试集ROC曲线",
            "image": os.path.join(save_model_dir, 'test_roc.png')
        }
    ]
    json.dump(metrics, open('/metric.json', mode='w'))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("train xgb launcher")
    arg_parser.add_argument('--train_dataset', type=str, help="训练数据集来源", default='')
    arg_parser.add_argument('--val_dataset', type=str, help="评估数据集名称", default='')
    arg_parser.add_argument('--feature_columns', type=str, help="特征列", default='')
    arg_parser.add_argument('--label_columns', type=str, help="标签列", default='')
    arg_parser.add_argument('--save_model_dir', type=str, help="模型地址", default='')
    arg_parser.add_argument('--inference_dataset', type=str, help="推理数据集名称", default='')
    arg_parser.add_argument('--model_params', type=str, help="模型的输入参数", default='{}')

    args = arg_parser.parse_args()
    if args.train_dataset:
        model_params = json.loads(args.model_params)
        train(save_model_dir=args.save_model_dir, train_dataset=args.train_dataset, label_columns=args.label_columns,
              feature_columns=args.feature_columns, model_params=model_params)
    if args.val_dataset:
        val(save_model_dir=args.save_model_dir, val_dataset=args.val_dataset, label_columns=args.label_columns,
            feature_columns=args.feature_columns)
    if args.inference_dataset:
        inference(save_model_dir=args.save_model_dir, inference_dataset=args.inference_dataset,
                  feature_columns=args.feature_columns)

'''
输入参数的写法
python xgb.py --save_model_dir /mnt/admin/pipeline/xgb/ --train_dataset /mnt/admin/data/data-test.csv --label_columns y --feature_columns age,duration,campaign,pdays,previous,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed --model_params '{"booster":"gbtree","eta":0.1,"gamma":0,"max_depth":6,"min_child_weight":1,"subsample":1,"colsample_bytree":1,"lambda":1,"objective":"binary:logistic","eval_metric":"auc"}'

python xgb.py --save_model_dir /mnt/admin/pipeline/xgb/ --val_dataset /mnt/admin/data/data-test.csv --label_columns y --feature_columns age,duration,campaign,pdays,previous,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed

python xgb.py --save_model_dir /mnt/admin/pipeline/xgb/ --inference_dataset /mnt/admin/data/data-test.csv --label_columns y --feature_columns age,duration,campaign,pdays,previous,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed
'''