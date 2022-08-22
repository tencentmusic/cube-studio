from collections import Counter

import config
from utils.utils import flatten_lists


class Metrics:
    """模型评价模块，计算每个标签的精确率、召回率、F1分数"""

    def __init__(self, golden_tags, predict_tags, remove_O=False) -> None:
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        # 不统计非实体标记
        if remove_O:
            pass

        # 统计tag总数
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()  # 每种tag 预测正确的数量
        self.predict_tags_count = Counter(self.predict_tags)
        self.golden_tags_count = Counter(self.golden_tags)

        # TP (True Positive) 真实-->0 预测-->0
        # FN (False Negative) 真实-->0 预测-->1
        # FP (False Positive) 真实-->1 预测-->0
        # TN (True Negative) 真实-->1 预测-->1

        # 精确率  p = len(gold_tag==predict_tag) / len(predict_tag)
        self.precision_scores = self.cal_precision()
        # 召回率  r = len(gold_tag==predict_tag) / len(gold_tag)
        self.recall_scores = self.cal_recall()
        # F1     F1 = (2*p*c) / (p+c)
        self.f1_scores = self.cal_f1()

    def count_correct_tags(self):
        """计算每个tag 预测正确的数量
        calculate the number of the every kind of tag that be predictded correctly"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag in correct_dict:
                    correct_dict[gold_tag] += 1
                else:
                    correct_dict[gold_tag] = 1
        return correct_dict

    def cal_precision(self):
        """计算每个标签的精确率"""
        precision_score = {}
        for tag in self.tagset:
            if tag not in self.correct_tags_number:
                precision_score[tag] = -1
                continue
            if tag not in self.predict_tags_count:
                precision_score[tag] = -2
                continue
            precision_score[tag] = self.correct_tags_number[tag] / self.predict_tags_count[tag] * 1.0
        return precision_score

    def cal_recall(self):
        """计算每个标签的召回率"""
        recall_score = {}
        for tag in self.tagset:
            if tag not in self.correct_tags_number:
                recall_score[tag] = -1
                continue
            if tag not in self.golden_tags_count:
                recall_score[tag] = -2
                continue
            recall_score[tag] = self.correct_tags_number[tag] / self.golden_tags_count[tag] * 1.0
        return recall_score

    def cal_f1(self):
        """计算f1分数"""
        f1_score = {}
        for tag in self.tagset:
            f1_score[tag] = 2 * self.precision_scores[tag] * self.recall_scores[tag] / (
                        self.precision_scores[tag] + self.recall_scores[tag])
        return f1_score

    def report_scores(self, dtype='HMM'):
        """将结果用表格的形式打印出来，像这个样子：

                      precision    recall  f1-score   support
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634

          avg/total      0.779     0.764     0.770      6178
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support', 'predict', '==']
        with open('result.txt', 'a') as fout:
            fout.write('\n')
            fout.write('=' * 100)
            fout.write('\n')
            fout.write('模型：{}，test结果如下：'.format(dtype))
            fout.write('\n')
            fout.write(header_format.format('', *header))
            print(header_format.format('', *header))

            # with open(config.data_path+'labels.txt') as f:
            #     tag_list = f.read().split()

            with open('./zdata/'+'labels.txt') as f:
                tag_list = f.read().split()

            row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9} {:>9} {:>9}'
            # 打印每个标签的 精确率、召回率、f1分数
            for tag in tag_list:
                if tag not in self.tagset: continue
                content = row_format.format(
                    tag,
                    self.precision_scores[tag],
                    self.recall_scores[tag],
                    self.f1_scores[tag],
                    self.golden_tags_count[tag],
                    self.predict_tags_count[tag],
                    self.correct_tags_number[tag] if tag in self.correct_tags_number else 0
                )
                fout.write('\n')
                fout.write(content)
                print(content)

            # 计算并打印平均值
            avg_metrics = self._cal_weighted_average()
            content = row_format.format(
                'avg/total',
                avg_metrics['precision'],
                avg_metrics['recall'],
                avg_metrics['f1_score'],
                len(self.golden_tags),
                len(self.predict_tags),
                sum(self.correct_tags_number.values())
            )
            fout.write('\n')
            fout.write(content)
            fout.write('\n')
            print(content)

    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)

        # 计算weighted precisions：
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_count[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average