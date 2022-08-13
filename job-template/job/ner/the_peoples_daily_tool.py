# 读取数据集
from re import S


with open('data/ThePeoplesDaily/raw/source_BIO_2014_cropus.txt', 'r') as source_file:
    source = source_file.read().split('\n')

with open('data/ThePeoplesDaily/raw/target_BIO_2014_cropus.txt', 'r') as target_file:
    target = target_file.read().split('\n')


# 统计每个样本的句长
max_len = 0
seq_len_dict = {}
for sentence in source:
    seq_len = (len(sentence) + 1) / 2
    if max_len < seq_len:
        max_len = seq_len
    if seq_len in seq_len_dict: seq_len_dict[seq_len] += 1
    else: seq_len_dict[seq_len] = 1
print(max_len)
seq_len_dict = sorted(seq_len_dict.items(), key=lambda x: x[0], reverse=True)
print(seq_len_dict)







# # 统计样本数量
# assert len(source) == len(target)
# sample_count = len(source)
# print('行数：', sample_count)

# # 统计字数 字典 标签种类
# vocabulary = dict()
# labels = dict()
# char_count = 0
# for sentence, tags in zip(source, target):
#     char_count += round( (len(sentence)+0.49)/2 )
#     for char, tag in zip(sentence.split(' '), tags.split(' ')):
#         if char in vocabulary: vocabulary[char] += 1
#         else: vocabulary[char] = 1
#         if tag in labels: labels[tag] += 1
#         else: labels[tag] = 1
# print('字数：', char_count)

# # 根据数量 降序 并写入vocabulary.txt和labels.txt文件
# vocabulary = dict(sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))
# labels = dict(sorted(labels.items(), key=lambda x: x[1], reverse=True))

# with open('data/ThePeoplesDaily/vocabulary.txt', 'w') as vocabulary_file:
#     vocabulary_file.write('\n'.join(vocabulary))

# with open('data/ThePeoplesDaily/labels.txt', 'w') as labels_file:
#     labels_file.write('\n'.join(labels))

# # 划分 训练集 验证集 测试集
# train_data_size = 0.7 
# dev_data_size = 0.2
# test_data_size = 0.1

# train_source, train_target = source[0 : round(train_data_size*sample_count)], target[0 : round(train_data_size*sample_count)]
# with open('data/ThePeoplesDaily/train.txt', 'w') as train_file:
#     for sentence, tags in zip(train_source, train_target):
#         for char, tag in zip(sentence.split(' '), tags.split(' ')):
#             train_file.write(char + ' ' + tag + '\n')
#         train_file.write('\n')
# print('train_data_sample_size: ', len(train_source))

# dev_source, dev_target = source[round(train_data_size*sample_count) : round((train_data_size+dev_data_size)*sample_count)], target[round(train_data_size*sample_count) : round((train_data_size+dev_data_size)*sample_count)]
# with open('data/ThePeoplesDaily/dev.txt', 'w') as dev_file:
#     for sentence, tags in zip(dev_source, dev_target):
#         for char, tag in zip(sentence.split(' '), tags.split(' ')):
#             dev_file.write(char + ' ' + tag + '\n')
#         dev_file.write('\n')
# print('dev_data_sample_size: ', len(dev_source))

# test_source, test_target = source[round((train_data_size+dev_data_size)*sample_count) : -1], target[round((train_data_size+dev_data_size)*sample_count) : -1]
# with open('data/ThePeoplesDaily/test.txt', 'w') as test_file:
#     for sentence, tags in zip(test_source, test_target):
#         for char, tag in zip(sentence.split(' '), tags.split(' ')):
#             test_file.write(char + ' ' + tag + '\n')
#         test_file.write('\n')
# print('test_data_sample_size: ', len(test_source))




with open('data/ThePeoplesDaily/raw/source_BIO.txt', 'w') as file:
    for sentence, tags in zip(source, target):
        for char, tag in zip(sentence.split(' '), tags.split(' ')):
            file.write(char + ' ' + tag + '\n')
        file.write('\n')
print('data_sample_size: ', len(source))