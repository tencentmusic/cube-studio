import shutil
# 加载相关的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import pysnooper
import os, sys
import argparse
import pandas, json

import os
import pickle
import xgboost as xgb


# 加载IMDb数据集
# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
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
def inference(load_model_path,save_model_dir, inference_dataset, feature_columns):
    if save_model_dir.endswith(".pkl"):
        os.makedirs(os.path.dirname(os.path.abspath(save_model_dir)),exist_ok=True)
    else:
        os.makedirs(save_model_dir,exist_ok=True)

    #兼容模型文件在哪里指定，以及指定的save_dir是模型文件还是文件夹的几种情况
    if os.path.isdir(save_model_dir) and load_model_path=='':
        load_model_path=os.path.join(save_model_dir,'xgb_model.pkl')
    elif not os.path.isdir(save_model_dir) and load_model_path=='':
        load_model_path = save_model_dir
        save_model_dir = os.path.dirname(os.path.abspath(save_model_dir))
    elif not os.path.isdir(save_model_dir) and load_model_path!='':
        save_model_dir = os.path.dirname(os.path.abspath(save_model_dir))
    
    #推理的y值根据xgb_label_mapping.json映射回原来的y
    mapping_path = os.path.join(save_model_dir,'xgb_label_mapping.json')
    label_mapping = json.load(open(mapping_path, "r")) if os.path.exists(mapping_path) else {}
    inverted_label_mapping = {value: key for key, value in label_mapping.items()}
        
    with open(load_model_path, 'rb') as file:
        model = pickle.load(file)
        df = pandas.read_csv(inference_dataset)
        feature_columns = [x.strip() for x in feature_columns.split(',') if x.strip()]
        X = df[feature_columns]  # 目标变量

        y_pred_prob = model.predict_proba(X)
        y_pred = model.predict(X)
        print('预测值')
        print(y_pred)
        y_df = pd.DataFrame(y_pred,columns=['y'])

        result = pd.concat([X,y_df],axis=1)
        result = pd.concat([result,pd.DataFrame(y_pred_prob)],axis=1)
        if inverted_label_mapping!={} and len(set(y_pred)-set([0,1]))!=0:
            result['y'] =  result['y'].apply(lambda x:int(inverted_label_mapping[str(x)]) if is_number(inverted_label_mapping[str(x)]) else inverted_label_mapping[str(x)])
            columns = list(result.columns)
            max_length = len(columns)-1
            for i in range(len(inverted_label_mapping.keys())):
                v = inverted_label_mapping[str(columns[max_length-i])]
                if is_number(v):
                    columns[max_length-i] = int(v)
                else:
                    columns[max_length-i] = str(v)
            result.columns = columns

        result.to_csv(os.path.join(save_model_dir,'inference_result.csv'), index=False, header=True)
        print('推理结果保存至', os.path.join(save_model_dir,'inference_result.csv'))


@pysnooper.snoop()
def val(load_model_path,save_model_dir, val_dataset, label_columns, feature_columns):
    if save_model_dir.endswith(".pkl"):
        os.makedirs(os.path.dirname(os.path.abspath(save_model_dir)),exist_ok=True)
    else:
        os.makedirs(save_model_dir,exist_ok=True)
        
    #兼容模型文件在哪里指定，以及指定的save_dir是模型文件还是文件夹的几种情况
    if os.path.isdir(save_model_dir) and load_model_path=='':
        load_model_path=os.path.join(save_model_dir,'xgb_model.pkl')
    elif not os.path.isdir(save_model_dir) and load_model_path=='':
        load_model_path = save_model_dir
        save_model_dir = os.path.dirname(os.path.abspath(save_model_dir))
    elif not os.path.isdir(save_model_dir) and load_model_path!='':
        save_model_dir = os.path.dirname(os.path.abspath(save_model_dir))
        
    save_val_path = os.path.join(save_model_dir,'val_result.json')
    mapping_path = os.path.join(save_model_dir,'xgb_label_mapping.json')
    label_mapping = json.load(open(mapping_path, "r")) if os.path.exists(mapping_path) else {}
    train_test = json.load(open(save_val_path, "r")) if os.path.exists(save_val_path) else {}
    
    with open(load_model_path, 'rb') as file:
        model = pickle.load(file)
        df = pandas.read_csv(val_dataset)
        label_columns = [x.strip() for x in label_columns.split(',') if x.strip()]
        feature_columns = [x.strip() for x in feature_columns.split(',') if x.strip()]

        X_val = df[feature_columns]  #
        y_val = df[label_columns]  # 目标变量
        
        if label_mapping!={} and len(set(y_val)-set([0,1]))!=0:
            y_val = df[label_columns[0]].apply(lambda x:int(label_mapping[str(x)]) if is_number(label_mapping[str(x)]) else label_mapping[str(x)])
        print(y_val,type(y_val[0]),y_val.unique())

        # dval = xgb.DMatrix(X_val, y_val)
        if len(set(y_val)-set([0,1]))==0:
            y_pred = model.predict_proba(X_val)[:,1]

            # 评估模型
            accuracy = accuracy_score(y_val, model.predict(X_val))
            print('预测值')
            print(y_pred)

            auc = roc_auc_score(y_val,y_pred)

            draw_roc_curve(y_val, y_pred,os.path.join(save_model_dir,'val_roc.png'))

            print('Accuracy:', accuracy)
            print('Auc:', auc)

            print(train_test)
            train_test.update({"val_acc":accuracy,"val_auc":auc})
            json.dump(train_test,open(save_val_path, "w"))
            print('评估结果保存至', save_val_path)
        else:
            y_pred = model.predict(X_val)

            val_accuracy = accuracy_score(y_val, y_pred)
            val_precision = precision_score(y_val, y_pred, average='weighted')  # 使用'macro'平均计算每个类别的精确度
            val_recall = recall_score(y_val, y_pred, average='weighted')  # 使用'macro'平均计算每个类别的召回率
            val_f1 = f1_score(y_val, y_pred, average='weighted')  # 使用'macro'平均计算每个类别的F1分数            
            train_test.update({"multi_class_val_acc":val_accuracy,"multi_class_val_prec":val_precision,"multi_class_val_recall":val_recall,"multi_class_val_f1":val_f1})
            json.dump(train_test,open(save_val_path, "w"))
            print('评估结果保存至', save_val_path)

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
    if save_model_dir.endswith(".pkl"):
        os.makedirs(os.path.dirname(os.path.abspath(save_model_dir)),exist_ok=True)
    else:
        os.makedirs(save_model_dir,exist_ok=True)

    #save_model_dir允许用户输入模型pkl文件名或文件夹，如果是文件名，使用其上层目录作为其他输出文件的保存目录
    if os.path.isdir(save_model_dir):
        save_model_path=os.path.join(save_model_dir,'xgb_model.pkl')
    else:
        save_model_path = save_model_dir
        save_model_dir = os.path.dirname(os.path.abspath(save_model_dir))
        
    # 读取数据
    # if '.csv' in dataset_path:
    df = pandas.read_csv(train_dataset)
    label_columns = [x.strip() for x in label_columns.split(',') if x.strip()]
    feature_columns = [x.strip() for x in feature_columns.split(',') if x.strip()]

    # 处理数据
    # X = df.drop(label_columns, axis=1)  # 特征变量
    X = df[feature_columns]  # 目标变量
    y = df[label_columns]

    # 使用LabelEncoder将y值转换为标准的数值型，也就是0,1,2,3等
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(y,type(y[0]))
    # 获取标签和编码的对应关系，并把对应关系保存下来
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(label_mapping)
    with open(os.path.join(save_model_dir,'xgb_label_mapping.json'), 'w', encoding='utf-8') as json_file:
        json.dump({str(k):str(v) for k, v in label_mapping.items()}, json_file, ensure_ascii=False, indent=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分训练集和测试集
    # 训练模型
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    
    if len(set(y)-set([0,1]))==0:
        #把训练集和测试集的y存下来，用于评估
        accuracy_train = accuracy_score(y_train, model.predict(X_train))
        accuracy_test = accuracy_score(y_test, model.predict(X_test))

        y_train_pred = model.predict_proba(X_train)[:,1]
        y_test_pred = model.predict_proba(X_test)[:,1]
        #计算auc
        auc_train = roc_auc_score(y_train, y_train_pred)
        auc_test = roc_auc_score(y_test, y_test_pred)

        draw_roc_curve(y_train, y_train_pred,os.path.join(save_model_dir,'train_roc.png'))
        draw_roc_curve(y_test, y_test_pred,os.path.join(save_model_dir,'test_roc.png'))

        # file = open(os.path.join(save_model_dir,'val_result.json'), "w")
        file = open(os.path.join(save_model_dir,'val_result'+json.dumps(model_params)+'.json'), "w")
        file.write(json.dumps({"train_acc":accuracy_train,"test_acc":accuracy_test,"train_auc":auc_train,"test_auc":auc_test}))
        file.close()
    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted')  # 使用'macro'平均计算每个类别的精确度
        train_recall = recall_score(y_train, y_train_pred, average='weighted')  # 使用'macro'平均计算每个类别的召回率
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')  # 使用'macro'平均计算每个类别的F1分数
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')  # 使用'macro'平均计算每个类别的精确度
        test_recall = recall_score(y_test, y_test_pred, average='weighted')  # 使用'macro'平均计算每个类别的召回率
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')  # 使用'macro'平均计算每个类别的F1分数

        file = open(os.path.join(save_model_dir,'val_result'+json.dumps(model_params)+'.json'), "w")
        file.write(json.dumps({"multi_class_train_acc":train_accuracy,"multi_class_test_acc":test_accuracy,"multi_class_train_prec":train_precision,"multi_class_test_prec":test_precision,"multi_class_train_recall":train_recall,"multi_class_test_recall":test_recall,"multi_class_train_f1":train_f1,"multi_class_test_f1":test_f1}))
        file.close()

    with open(save_model_path, 'wb') as file:
        pickle.dump(model, file)
        print('训练模型导出至',save_model_path)
        
# 原生的xgb的代码，原生xgb需要通过"multi:softprob"和"multi:softmax"来输出概率和分类
#     dtrain = xgb.DMatrix(X_train, y_train)
#     dtest = xgb.DMatrix(X_test, y_test)

#     evallist = [(dtrain, 'train'), (dtest, 'test')]

#     #计算输出为prob的xgb模型
#     model_params['objective'] = "multi:softprob"
#     # 训练xgb模型
#     model = xgb.train(model_params, dtrain, evals=evallist)
#     # 计算prob
#     y_train_pred_prob = model.predict(dtrain)
#     y_test_pred_prob = model.predict(dtest)
#     with open(save_model_path.replace('xgb_model.pkl','xgb_model_prob.pkl'), 'wb') as file:
#         pickle.dump(model, file)
    
#     #计算输出为分类的xgb模型
#     model_params['objective'] = "multi:softmax"
#     # 训练xgb模型
#     model = xgb.train(model_params, dtrain, evals=evallist)
#     # 计算分类
#     y_train_pred_class = model.predict(dtrain)
#     y_test_pred_class = model.predict(dtest)
#     with open(save_model_path.replace('xgb_model.pkl','xgb_model_class.pkl'), 'wb') as file:
#         pickle.dump(model, file)
    
#     if len(set(y[label_columns[0]].unique())-set([0,1]))==0:
#         y_train_pred_prob = pd.DataFrame(y_train_pred_prob).iloc[:,1]
#         y_test_pred_prob = pd.DataFrame(y_test_pred_prob).iloc[:,1]

#         # 计算auc
#         auc_train = roc_auc_score(y_train, y_train_pred_prob)
#         auc_test = roc_auc_score(y_test, y_test_pred_prob)

#         # 计算acc
#         train_acc = accuracy_score(y_train, y_train_pred_class)
#         test_acc = accuracy_score(y_test, y_test_pred_class)

#         draw_roc_curve(y_train, y_train_pred_prob, os.path.join(save_model_dir, 'train_roc.png'))
#         draw_roc_curve(y_test, y_test_pred_prob, os.path.join(save_model_dir, 'test_roc.png'))

#         file = open(os.path.join(save_model_dir, 'val_result' + json.dumps(model_params) + '.json'), "w")
#         file.write(json.dumps({"train_acc": train_acc,"train_auc": auc_train, "test_acc": test_acc,"test_auc": auc_test}))
#         file.close()
#     else:        
#         train_accuracy = accuracy_score(y_train, y_train_pred_class)
#         train_precision = precision_score(y_train, y_train_pred_class, average='weighted')  # 使用'macro'平均计算每个类别的精确度
#         train_recall = recall_score(y_train, y_train_pred_class, average='weighted')  # 使用'macro'平均计算每个类别的召回率
#         train_f1 = f1_score(y_train, y_train_pred_class, average='weighted')  # 使用'macro'平均计算每个类别的F1分数
        
#         test_accuracy = accuracy_score(y_test, y_test_pred_class)
#         test_precision = precision_score(y_test, y_test_pred_class, average='weighted')  # 使用'macro'平均计算每个类别的精确度
#         test_recall = recall_score(y_test, y_test_pred_class, average='weighted')  # 使用'macro'平均计算每个类别的召回率
#         test_f1 = f1_score(y_test, y_test_pred_class, average='weighted')  # 使用'macro'平均计算每个类别的F1分数

#         file = open(os.path.join(save_model_dir,'val_result'+json.dumps(model_params)+'.json'), "w")
#         file.write(json.dumps({"multi_class_train_acc":train_accuracy,"multi_class_test_acc":test_accuracy,"multi_class_train_prec":train_precision,"multi_class_test_prec":test_precision,"multi_class_train_recall":train_recall,"multi_class_test_recall":test_recall,"multi_class_train_f1":train_f1,"multi_class_test_f1":test_f1}))
#         file.close()


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
    arg_parser.add_argument('--train_dataset', type=str, help="训练数据集地址", default='')
    arg_parser.add_argument('--val_dataset', type=str, help="评估数据集地址", default='')
    arg_parser.add_argument('--feature_columns', type=str, help="特征列", default='')
    arg_parser.add_argument('--label_columns', type=str, help="标签列", default='')
    arg_parser.add_argument('--save_model_dir', type=str, help="模型地址", default='')
    arg_parser.add_argument('--inference_dataset', type=str, help="推理数据集地址", default='')
    arg_parser.add_argument('--model_params', type=str, help="模型的输入参数", default='{}')
    arg_parser.add_argument('--load_model_path', type=str, help="加载模型地址", default='')

    args = arg_parser.parse_args()
    if args.train_dataset:
        model_params = json.loads(args.model_params)
        train(save_model_dir=args.save_model_dir, train_dataset=args.train_dataset, label_columns=args.label_columns,
              feature_columns=args.feature_columns, model_params=model_params)
    if args.val_dataset:
        val(load_model_path=args.load_model_path,save_model_dir=args.save_model_dir, val_dataset=args.val_dataset, label_columns=args.label_columns,
            feature_columns=args.feature_columns)
    if args.inference_dataset:
        inference(load_model_path=args.load_model_path,save_model_dir=args.save_model_dir, inference_dataset=args.inference_dataset,
                  feature_columns=args.feature_columns)

'''
输入参数的写法
python xgb.py --save_model_dir /mnt/admin/pipeline/xgb/ --train_dataset /mnt/admin/data/data-test.csv --label_columns y --feature_columns age,duration,campaign,pdays,previous,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed --model_params '{"booster":"gbtree","eta":0.1,"gamma":0,"max_depth":6,"min_child_weight":1,"subsample":1,"colsample_bytree":1,"lambda":1,"objective":"binary:logistic","eval_metric":"auc"}'

python xgb.py --save_model_dir /mnt/admin/pipeline/xgb/ --val_dataset /mnt/admin/data/data-test.csv --label_columns y --feature_columns age,duration,campaign,pdays,previous,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed

python xgb.py --save_model_dir /mnt/admin/pipeline/xgb/ --inference_dataset /mnt/admin/data/data-test.csv --label_columns y --feature_columns age,duration,campaign,pdays,previous,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed
'''
