class Preprocessing:
    """
    数据预处理：
        划分 训练集、验证集、测试集
        构建 字典
    """

    def __init__(self, file_path: str, file_name: str) -> None:
        self.file_path = file_path
        self.file_name = file_name
        with open(file_path + file_name, 'r') as file:
            self.item_list = file.read().split('\n\n')
        print('样本数量：', len(self.item_list))
    
    def train_test_dev_split(self, data_rate: list):
        assert len(data_rate)==3 and sum(data_rate)==1

        # 划分 训练集 验证集 测试集
        train_data_size = data_rate[0] 
        dev_data_size = data_rate[1]
        test_data_size = data_rate[2]

        train_data = self.item_list[0 : round(train_data_size*len(self.item_list))]
        with open(f'{self.file_path}train.txt', 'w') as train_file:
            train_file.write('\n\n'.join(train_data))
            train_file.write('\n')
        print('train_data_sample_size: ', len(train_data))

        dev_data = self.item_list[round(train_data_size*len(self.item_list)) : round((train_data_size+dev_data_size)*len(self.item_list))]
        with open(f'{self.file_path}dev.txt', 'w') as dev_file:
            dev_file.write('\n\n'.join(dev_data))
            dev_file.write('\n')
        print('dev_data_sample_size: ', len(dev_data))

        test_data = self.item_list[round((train_data_size+dev_data_size)*len(self.item_list)) : -1]
        with open(f'{self.file_path}test.txt', 'w') as test_file:
            test_file.write('\n\n'.join(test_data))
            test_file.write('\n')
        print('test_data_sample_size: ', len(test_data))

    def construct_vocabulary_labels(self):
        
        char_count = 0
        vocabulary = dict()
        labels = dict()
        
        for item in self.item_list:
            for char_tag in item.split('\n'):
                try:
                    char = char_tag.split(' ')[0]
                    tag = char_tag.split(' ')[1]
                    if char in vocabulary: vocabulary[char] += 1
                    else: vocabulary[char] = 1
                    if tag in labels: labels[tag] += 1
                    else: labels[tag] = 1
                    char_count += 1
                except:
                    print(char_tag)
        print('字数：', char_count)

        # 根据数量 降序 并写入vocabulary.txt和labels.txt文件
        vocabulary = dict(sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))
        labels = dict(sorted(labels.items(), key=lambda x: x[1], reverse=True))

        with open(f'{self.file_path}vocabulary.txt', 'w') as vocabulary_file:
            vocabulary_file.write('\n'.join(vocabulary))
        print('vocabulary.txt constructed')

        with open(f'{self.file_path}labels.txt', 'w') as labels_file:
            labels_file.write('\n'.join(labels))
        print('labels.txt constructed')


if __name__ == '__main__':
    p = Preprocessing(file_path='./zdata/', file_name='annotated_data.txt')
    p.train_test_dev_split(data_rate=[0.7, 0.1, 0.2])
    p.construct_vocabulary_labels()