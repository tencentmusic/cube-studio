# from ..data import build_corpus

# files = ['train', 'dev', 'test']

# words = ['PAD']
# for file_name in files:
#     with open(f'./data/{file_name}.char', 'r') as file_object:
#         words_tags = file_object.read().split('\n')
#     for word_tag in words_tags:
#         if word_tag is None: continue
#         word = word_tag.split(' ')[0]
#         if word not in words:
#             words.append(word)

# # print(words)

# with open('data/vocabulary.char', 'w') as file_object:
#     for word in words:
#         file_object.write(f'{word}\n')







import os

def build_corpus(split, make_vocab=True, data_dir='./data'):
    """数据读取
    """

    assert split.lower() in ["train", "dev", "test"]

    word_lists = []
    tag_lists = []
    with open(os.path.join(data_dir, split + '.char'), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
    
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps



# 查找序列最大值
# train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
# dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
# test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

# max_length = 0
# for word_lists in [train_word_lists, dev_word_lists, test_word_lists]:
#     for word_list in word_lists:
#         if len(word_list) > max_length:
#             max_length = len(word_list)
# print(max_length)




# train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
# dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
# test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

# seq_len_frequence = dict()

# for word_lists in [train_word_lists, dev_word_lists, test_word_lists]:
#     for word_list in word_lists:
#         if len(word_list) in seq_len_frequence:
#             seq_len_frequence[len(word_list)] += 1
#         else:
#             seq_len_frequence[len(word_list)] = 1
# print(seq_len_frequence)

# seq_len_frequence = sorted(seq_len_frequence.items(), key=lambda x: x[0])
# print(seq_len_frequence)

# print(dict(seq_len_frequence))
# X = dict(seq_len_frequence).keys()
# Y = dict(seq_len_frequence).values()

import matplotlib.pyplot as plt

# a = plt.figure(figsize=(5,10))
# plt.bar(x=X, height=Y)
# plt.show()

import pandas as pd
import numpy as np

df2 = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
df2.plot.bar()

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3]);  # Plot some data on the axes.