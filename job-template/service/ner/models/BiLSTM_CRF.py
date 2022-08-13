import torch
import torch.nn as nn

from models.BiLSTM import BiLSTM


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tagset_size, config, tag2id, dropout=0.1, batch_first=True):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = config.input_size
        self.hidden_dim = config.hidden_size
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        # self.vocabulary = word2id
        self.tag2id = tag2id

        self.bilstm = BiLSTM(
            vocab_size=vocab_size,
            tagset_size=tagset_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        self.transition = nn.Parameter(
            # torch.randn(self.tagset_size, self.tagset_size)
            # torch.ones(self.tagset_size, self.tagset_size)
            torch.ones(self.tagset_size, self.tagset_size) * 1 / self.tagset_size
        )


    def forward(self, sentences):
        # B, L, out_size(tagset_size)
        emission =  self.bilstm._get_lstm_features(sentences)

        # calculate CRF scores 这个scores的大小为[B, L, out_size, out_size]
        # every Chinese Character map to a matrix of [tagset_size, tagset_size]
        # 该矩阵中的第i行 第j列的元素含义为：上一时刻tag为i，这一时刻tag为j的分数
        crf_scores = emission.unsqueeze(2).expand(-1, -1, self.tagset_size, -1) + self.transition.unsqueeze(0)

        return crf_scores

    def viterbi_decoding(self, crf_score):
        """viterbi decoding
        不支持 batch"""
        start_id = self.tag2id['<start>']
        end_id = self.tag2id['<end>']

        device = crf_score.device
        seq_len = crf_score.shape[0]
        viterbi = torch.zeros(seq_len, self.tagset_size).to(device)
        backpointer = (torch.ones(size=(seq_len, self.tagset_size), dtype=torch.long) * end_id).to(device)

        for step in range(seq_len):
            if step == 0:   # 第一个字
                viterbi[step, :] = crf_score[step, start_id, :]
                backpointer[step, :] = start_id
            else:
                max_scores, prev_tags_id = torch.max(
                    viterbi[step-1, :].unsqueeze(1) + crf_score[step, :, :],
                    dim = 0
                )
                viterbi[step, :] = max_scores
                backpointer[step, :] = prev_tags_id

        best_end_idx = end_id
        best_path = []
        for step in range(seq_len-1, 0, -1):
            if step == seq_len-1:
                best_path.append(backpointer[step, best_end_idx].item())
            else:
                best_path.append(backpointer[step, best_path[-1]].item())
        best_path.reverse()

        return best_path

    def loss(self, crf_scores, targets):
        """计算双向LSTM-CRF模型的损失
        以 batch 形式输入"""

        device = crf_scores.device

        pad_id = self.tag2id.get('<pad>')
        start_id = self.tag2id.get('<start>')
        end_id = self.tag2id.get('<end>')

        mask = (targets != pad_id)
        batch_size, max_len = targets.size()
        lengths = mask.sum(dim=1)

        # Golden scores 正确标签得分的和
        #     crf_scores矩阵形为[batch_size, max_len, tagset_size, tagset_size]
        #     crf_scores的第一个维度以句子为单位 crf_scores的第二个维度以单个字为单位
        #     每个字由一个[i=tagset_size, j=tagset_size]矩阵表示，该矩阵的具体含义为
        #     在前一个字的标签为i的前提下，当前字的标签为j的概率
        # 即 golden_socres 的值为每一个字矩阵中正确[i, j]下标的概率值之和
        
        # target           tagset_size*target_size的矩阵  保存该字正确的tag索引
        # former_target    tagset_size*target_size的矩阵  保存该字前一个字正确的tag索引
        former_targets = torch.zeros_like(targets)
        former_targets[:, 0] = start_id
        former_targets[:, 1:max_len] = targets[:, 0:max_len-1]

        # 根据当前字正确的tag索引抽取  即 j
        crf_scores_j = crf_scores.gather(
            dim=3,
            index=targets.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.tagset_size, -1)
        )
        # 根据前一个字正确的tag索引抽取  即 i
        crf_score_i_j = crf_scores_j.gather(
            dim=2,
            index=former_targets.unsqueeze(-1).unsqueeze(-1)
        )
        # 去除 pad, 并求和
        gold_score = crf_score_i_j.masked_select(mask.unsqueeze(-1).unsqueeze(-1)).sum()


        # 计算所有可能的值的和
        current_scores = torch.zeros(batch_size, self.tagset_size).to(device)
        for step in range(max_len):
            # 当前时刻 有效的batch_size（因为有些序列比较短)
            batch_size_step = (lengths > step).sum().item()
            if step == 0:
                current_scores[:batch_size_step] = crf_scores[:batch_size_step, step, start_id, :]
            else:
                # We add scores at current timestep to scores accumulated up to previous
                # timestep, and log-sum-exp Remember, the cur_tag of the previous
                # timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores
                # along cur. timestep's cur_tag dimension
                current_scores[:batch_size_step] = torch.logsumexp(
                    current_scores[:batch_size_step].unsqueeze(2) + crf_scores[:batch_size_step, step, :, :],
                    dim=1
                )
        all_scores = current_scores[:, end_id].sum()

        loss = (all_scores - gold_score) / batch_size
        return loss
