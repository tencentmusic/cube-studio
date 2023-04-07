'''
Author: your name
Date: 2021-06-16 10:18:30
LastEditTime: 2021-06-23 11:58:26
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /job-template/job/model_template/tf/ple/data_prepare.py
'''
import tensorflow as tf
from tqdm import tqdm
import json

SHIFT = 5
FEATURE_LENGTH = SHIFT + 328 - 1

def modify_head(head):
    new_head = head.replace('#', '_') # avoid fail to set placeholder name
    return new_head

def parse_csv_line_for_personal_radio(header, line, sep=' '):
    v = line.split(sep)
    if len(v) < FEATURE_LENGTH: #特征不足
        return None
    header = [modify_head(h) for h in header]
    feature_dict = {
        'ctr_label': tf.train.Feature(float_list=tf.train.FloatList(value=[float(v[2])])),
        'cvr_label': tf.train.Feature(float_list=tf.train.FloatList(value=[float(v[3])])),
        'like_label': tf.train.Feature(float_list=tf.train.FloatList(value=[float(v[4])]))
    }
    for i in range(SHIFT, FEATURE_LENGTH):
        if header[i] == 'albumHash':
            feature_dict[header[i]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v[i])])) # album hash embedding
        else:
            feature_dict[header[i]] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(v[i])]))
    features = tf.train.Features(
        feature = feature_dict
    )
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    return serialized

def txt_file_to_tfrecord(head_fn, txt_fn, tfrecord_prefix_fn, sep=' '):
    tfrecord_fn = tfrecord_prefix_fn + '.zip'
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    with open(txt_fn) as ft, open(head_fn) as fh, tf.io.TFRecordWriter(tfrecord_fn, options) as writer:
        header = None
        for line in fh:
            if line:
                line = line.strip().split(sep)[: FEATURE_LENGTH]
                if len(line) == FEATURE_LENGTH:
                    header = line
                break
        for line in tqdm(ft):
            if line:
                serialized = parse_csv_line_for_personal_radio(header, line, sep)
                writer.write(serialized)

def generate_input_json(header, label_names=[], embedding_names=[], eliminate_names=[]):
    rst = dict()
    rst['inputs'] = generate_list_json(header, label_names, embedding_names, eliminate_names, False)
    return rst

def generate_list_json(header, label_names=[], embedding_names=[], eliminate_names=[], is_tfrecord=True):
    header = [modify_head(h) for h in header]
    inputs = []
    for name in header:
        if name not in embedding_names and name not in label_names and name not in eliminate_names:
            d = dict()
            d['name'] = name
            if is_tfrecord:
                d['ftype'] = 'fixlen'
            d['dtype'] = 'float32'
            inputs.append(d)
    for embedding_name in embedding_names:
        d = dict()
        d['name'] = embedding_name
        if is_tfrecord:
            d['ftype'] = 'fixlen'
        d['dtype'] = 'int64'
        if not is_tfrecord:
            d['vocab_size'] = 300000 + 2
            d['embedding_dim'] = 32
        inputs.append(d)
    for label_name in label_names:
        d = dict()
        d['name'] = label_name
        if is_tfrecord:
            d['ftype'] = 'fixlen'
        d['dtype'] = 'float32'
        if not is_tfrecord:
            d['is_label'] = True
        inputs.append(d)
    return inputs

def generate_json_f(head_fn, json_fn, is_tfrecord=False, sep=' '):
    with open(head_fn) as fh, open(json_fn, 'w') as fj:
        header = None
        for line in fh:
            if line:
                line = line.strip().split(sep)[: FEATURE_LENGTH]
                if len(line) == FEATURE_LENGTH:
                    header = line
                break
        if is_tfrecord:
            js = generate_list_json(header, label_names=['ctr_label', 'cvr_label'], embedding_names=['albumHash'], eliminate_names=['uin', 'songid', 'like_label'])
        else:
            js = generate_input_json(header, label_names=['ctr_label', 'cvr_label'], embedding_names=['albumHash'], eliminate_names=['uin', 'songid', 'like_label'])
        js_str = json.dumps(js)
        fj.write(js_str)

def generate_tfrecord_json_f(head_fn, json_fn, sep=' '):
    generate_json_f(head_fn, json_fn, True, sep)

def extract_headers(head_fn, out_fn, sep=' ', out_sep=','):
    with open(head_fn) as fh, open(out_fn, 'w') as fo:
        header = None
        for line in fh:
            if line:
                arr = line.strip().split(sep)
                arr = [e.replace('#', '_') for e in arr]
                arr_str = out_sep.join(arr)
                fo.write(arr_str)
                break

def split_train_val(data_fn, train_fn, val_fn):
    with open(data_fn) as fd, open(train_fn, 'w') as ft, open(val_fn, 'w') as fv:
        cnt = 0
        for line in fd:
            if line:
                if cnt % 10 == 0:
                    fv.write(line)
                else:
                    ft.write(line)
                cnt += 1

def clean_data(data_fn, out_fn):
    with open(data_fn) as fd, open(out_fn, 'w') as fo:
        cnt = 0
        for line in fd:
            cnt += 1
            if cnt % 100000 == 0:
                print(cnt)            
            arr = line.strip().split(' ')
            if len(arr) == 336:
                ok = True
                for i in range(5, 336):
                    try:
                        t = float(arr[i])
                    except:
                        ok = False
                        print('error item is: ' + arr[i])
                        break
                if ok:
                    fo.write(line)

if __name__ == '__main__':
    #txt_file_to_tfrecord('/Users/jurluo/myJob/ml/personal_radio/data/test.dat.norm.s', '/Users/jurluo/myJob/ml/personal_radio/data/test.dat.norm.shuf.s1w', '/Users/jurluo/myJob/ml/personal_radio/data/personalRadio')
    #generate_tfrecord_json_f('/Users/jurluo/myJob/ml/personal_radio/data/test.dat.norm.s', '/Users/jurluo/myJob/ml/personal_radio/data/tfr.json')
    #generate_json_f('/Users/jurluo/myJob/ml/personal_radio/data/test.dat.norm.s', '/Users/jurluo/myJob/ml/personal_radio/data/pr.json')
    #extract_headers('/Users/jurluo/myJob/ml/personal_radio/data/test.dat.norm.s', '/Users/jurluo/myJob/ml/personal_radio/data/header.txt')
    #split_train_val('/Users/jurluo/myJob/ml/personal_radio/data/test.dat.norm.shuf.s1w', '/Users/jurluo/myJob/ml/personal_radio/data/train.csv', '/Users/jurluo/myJob/ml/personal_radio/data/val.csv')
    clean_data('/Users/jurluo/myJob/ml/personal_radio/data/train.dat.norm.shuf', '/Users/jurluo/myJob/ml/personal_radio/data/train1.csv')