import pandas as pd
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from job.pkgs.utils import make_abs_or_pack_path, recur_expand_param
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
import random
from numpy import savetxt
import logging

import argparse
import json
import os
import sys
import numpy as np

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.DEBUG)

def gen_data_set(data, negsample=0):

    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]),rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0,len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i],1,len(hist[::-1]),rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]),len(test_set[0]))

    return train_set,test_set


def gen_model_input(train_set,user_profile,seq_max_len):

    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


def run_youtube_dnn(input_file, user_embed_output_file, item_embed_output_file):

    data = pd.read_csvdata = pd.read_csv(input_file)
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip"]
    SEQ_LEN = 50

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)

    train_set, test_set = gen_data_set(data, 0)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    embedding_dim = 16

    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            SparseFeat("gender", feature_max_idx['gender'], embedding_dim),
                            SparseFeat("age", feature_max_idx['age'], embedding_dim),
                            SparseFeat("occupation", feature_max_idx['occupation'], embedding_dim),
                            SparseFeat("zip", feature_max_idx['zip'], embedding_dim),
#                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
#                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    # 3.Define Model and train

    K.set_learning_phase(True)
    import tensorflow as tf
    if tf.__version__ >= '2.0.0':
       tf.compat.v1.disable_eager_execution()

    model = YoutubeDNN(user_feature_columns, item_feature_columns, 
        num_sampled=5, 
        user_dnn_hidden_units=(64, embedding_dim))

    model.compile(optimizer="adam", loss=sampledsoftmaxloss)  # "binary_crossentropy")

    history = model.fit(train_model_input, train_label,  # train_label,
                        batch_size=256, epochs=1, verbose=1, validation_split=0.0, )

    # 4. Generate user features for testing and full item features for retrieval
    test_user_model_input = test_model_input
    all_item_model_input = {"movie_id": item_profile['movie_id'].values}

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)
    print(item_embs.shape)
    
    savetxt(user_embed_output_file, user_embs, delimiter=',')
    savetxt(item_embed_output_file, item_embs, delimiter=',')
    

def main(job, pack_path, upstream_output_file, export_path):
    if job.get('--job'):
        job = job.get('--job')
        print("Renew job: {}".format(job))

    job_detail = job.get('job_detail')
    
    if not job_detail:
        print ('job_detail not set')
        return

    train_data_args = job_detail.get('train_data_args')
    if not train_data_args:
        print ('train_data_args not set')
        return 
    input_file = train_data_args.get('file')

    if input_file and input_file.strip():
        input_file = os.path.join(pack_path, os.path.basename(input_file.strip()))
    else:
        input_file = None
        print("input_file not set\n")
        return

    logging.info('input_file=%s'%str(input_file))

    user_embed_output_file = os.path.join(export_path, "user_embed.txt")
    item_embed_output_file = os.path.join(export_path, "item_embed.txt")
    
    logging.info('user_embed_output_file=%s', str(user_embed_output_file))
    logging.info('item_embed_output_file=%s', str(item_embed_output_file))

    run_youtube_dnn(input_file, user_embed_output_file, item_embed_output_file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser("Youtube DNN model runner train component")
    arg_parser.add_argument('--job', type=str, required=True, help="模型训练任务描述json")
    arg_parser.add_argument('--pack-path', type=str, required=True, help="用户包（包含所有用户文件的目录）的挂载到容器中的路径")
    arg_parser.add_argument('--upstream-output-file', type=str, help="上游输出文件（包含路径）")
    arg_parser.add_argument('--export-path', type=str, required=True, help="数据导出目录")

    args = arg_parser.parse_args()
    print("{} args: {}".format(__file__, args))

    job_spec = json.loads(args.job)
    print("job str: {}\n".format(args.job))
    job_spec = recur_expand_param(job_spec, args.export_path, args.pack_path)
    print("expanded job spec: {}\n".format(job_spec))
    main(job_spec, args.pack_path, args.upstream_output_file, args.export_path)