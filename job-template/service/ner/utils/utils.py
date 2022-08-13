import os

from operator import index
import pickle

def save_model(model, file_name):
    print('utils_work_path:',os.getcwd())
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)

def flatten_lists(lists):
    """将 list of list 展开为 list"""
    flatten_list = []
    for list_ in lists:
        if type(list_) == list:
            flatten_list.extend(list_)
        else:
            flatten_list.append(list_)
    return flatten_list

def expand_vocabulary(word2id, tag2id, crf=False, bert=False):
    """添加 <pad> <unk> 标签"""
    word2id['<pad>'] = len(word2id)
    word2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    tag2id['<unk>'] = len(tag2id)

    if crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(tag2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    if bert:
        word2id['<cls>'] = len(word2id)
        word2id['<seq>'] = len(word2id)
        tag2id['<cls>'] = len(tag2id)
        tag2id['<seq>'] = len(tag2id)

    return word2id, tag2id

def add_end_tag(word_lists, tag_lists=None):
    """用于bilstm_crf的训练数据需要的在句尾添加<end>标签, 添加 <end> 标签"""
    if tag_lists:
        assert len(word_lists) == len(tag_lists)
        for idx in range(len(word_lists)):
            word_lists[idx].append('<end>')
            tag_lists[idx].append('<end>')
        return word_lists, tag_lists
    else:
        for idx in range(len(word_lists)):
            word_lists[idx].append('<end>')
        return word_lists

def expand_4_bert(target):
    # tag2id['<cls>'] = len(tag2id)
    # tag2id['<seq>'] = len(tag2id)
    for idx, _ in enumerate(target):
        target[idx].insert(0, '<cls>')
        target[idx].append('<seq>')
    return target
