import torch


class HMM:

    def __init__(self, N, M) -> None:
        """
        :param N  状态数 --> 标签数量
        :param M  观测数 --> 词表数量
        """
        self.N = N
        self.M = M

        self.Pi = torch.zeros(size=(self.N,))  # 初始概率分布
        self.A = torch.zeros(size=(self.N, self.N))  # 状态转移概率分布
        self.B = torch.zeros(size=(self.N, self.M))  # 观测概率分布

    def train(self, word_lists, tag_lists, word2id, tag2id):
        """训练算法：根据训练数据学习HMM模型的参数 Pi、 A、 B
        训练数据中包含 S 个长度相同的观测序列和对应的状态
        序列 {(O1, I1), (O2, I2), ...},可以利用极大似然
        估计法来估计隐马尔可夫模型的参数
        
        :param word_lists  嵌套列表，[[高, 勇, ：, 男, ，, 中, 国, 国, 籍], [...], ... ]
        :param tag_lists   嵌套列表，[[B-NAME, E-NAME, O, O, O, B-CONT, M-CONT, M-CONT, E-CONT], [...], ...]
        :param word2id     词典，词汇表（以 单个汉字 为颗粒度）
        :param tag2id      字典，标签表
        """
        assert len(word_lists) == len(tag_lists)

        # Pi 估计初始概率分布
        for tag_list in tag_lists:
            # 统计 累加
            tag_idx = tag2id[tag_list[0]]
            self.Pi[tag_idx] += 1
        # smoth 平滑
        self.Pi[self.Pi == 0] = 1e-10
        # 计算概率
        self.Pi = self.Pi / self.Pi.sum()

        # A 估计状态转移概率分布
        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tag_idx = tag2id[tag_list[i]]
                next_tag_idx = tag2id[tag_list[i + 1]]
                self.A[current_tag_idx][next_tag_idx] += 1
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / torch.sum(self.A, dim=1, keepdim=True)

        # B 估计观测概率分布
        for word_list, tag_list in zip(word_lists, tag_lists):
            for word, tag in zip(word_list, tag_list):
                word_idx = word2id[word]
                tag_idx = tag2id[tag]
                self.B[tag_idx][word_idx] += 1
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / torch.sum(self.B, dim=1, keepdim=True)

        # print('N', self.N)
        # print('M', self.M)
        # print('Pi', self.Pi)
        # print('A', self.A)
        # print('B', self.B)

    def _viterbi_decoding(self, word_list, word2id, tag2id):

        # 将 Pi A B 取对数，可以将乘风转换为加法，
        # log(ab) == log(a) + log(b)
        Pi = torch.log(self.Pi)
        A = torch.log(self.A)
        B = torch.log(self.B)

        # viterbi矩阵      --> 《统计学习》例10.3中的 lambda函数
        # backpointer矩阵  --> 《统计学习》例10.3中的 psi函数
        seq_len = len(word_list)
        viterbi_matrix = torch.zeros(size=(self.N, seq_len))
        backpointer = torch.zeros(size=(self.N, seq_len)).long()

        # 初始化
        word_idx = word2id.get(word_list[0], None)
        Pit = Pi.t()
        if word_idx is None:
            b = torch.ones(size=(self.N,)) / self.N
        else:
            b = B[:, word_idx]
        viterbi_matrix[:, 0] = Pit + b
        backpointer[:, 0] = -1

        # 递推
        for step in range(1, seq_len):
            word_idx = word2id.get(word_list[step], None)
            for tag_idx in range(self.N):
                if word_idx is None:
                    b = (torch.ones(size=(self.N,)) / self.N)[tag_idx]
                else:
                    b = B[tag_idx, word_idx]
                P_former = viterbi_matrix[:, step - 1] + A[:, tag_idx]
                viterbi_matrix[tag_idx, step] = torch.max(P_former) + b
                backpointer[tag_idx, step] = torch.argmax(P_former)

        # 结束
        best_prob, best_path_end = torch.max(viterbi_matrix[:, -1]), torch.argmax(viterbi_matrix[:, -1])

        # 回溯
        best_path = []
        best_path.append(int(best_path_end))
        for step_ in range(seq_len - 1, 0, -1):
            # print(best_path[-1], step_)
            best_path.append(int(backpointer[best_path[-1], step_]))
        best_path.reverse()

        return best_path

    def test(self, word_lists, word2id, tag2id):
        pre_tag_lists = []
        id2tag = dict((id, tag) for tag, id in tag2id.items())
        for word_list in word_lists:
            predict_path = self._viterbi_decoding(word_list, word2id, tag2id)

            # 将tag_id组成的best_path转换为对应的tag
            assert len(predict_path) == len(word_list)
            predict_tag = [id2tag[tag_id] for tag_id in predict_path]

            pre_tag_lists.append(predict_tag)
        return pre_tag_lists

    def __call__(self, word_lists, word2id, tag2id):
        print(word_lists)
        return self.test(word_lists, word2id, tag2id)
