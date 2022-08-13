import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BiLSTM import BiLSTM
from config import TrainingConfig, BiLSTMConfig

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

    def forward(self, sentence):
        # B, L, out_size(tagset_size)
        bilstm_score =  self.bilstm._get_lstm_features(sentence)

        # calculate CRF scores 这个scores的大小为[B, L, out_size, out_size]
        # every Chinese Character map to a matrix of [tagset_size, tagset_size]
        # 该矩阵中的第i行 第j列的元素含义为：上一时刻tag为i，这一时刻tag为j的分数
        crf_scores = bilstm_score.unsqueeze(2).expand(-1, -1, self.tagset_size, -1) + self.transition.unsqueeze(0)

        return crf_scores

    def neg_log_likelihood_iteration(self, scores, targets):
        """迭代计算crf loss
        此方法一次计算一个样本的loss
        一个batch的loss需要迭代batch_size次
        消耗的计算时长较多

        crf loss 负对数似然损失
        loss        = exp(gold_path) / sum(exp(all_path))
        log(loss)   = log(exp(gold_path)) - log(sum(exp(all_path)))
                    = gold_path_score - all_path_score
        -log(loss)  = all_path_score = gold_path)score"""
        assert len(scores) == len(targets)
        losses = 0.
        for score, target in zip(scores, targets):
            # 一个batch由多个sentence的score组成
            
            assert len(score) == len(target)
            PAD_ID = self.tag2id['<pad>']
            START_ID = self.tag2id['<start>']
            END_ID = self.tag2id['<end>']
            device = target.device
            mask = (target != PAD_ID)
            current_target = target.masked_select(mask)
            former_target = torch.cat(tensors=(torch.LongTensor([START_ID]).to(device), current_target[0:-1]))
            seq_len = len(current_target)

            gold_path_score = 0.
            for step in range(seq_len):
                # gold_path_score
                gold_path_score += score[step, former_target[step], current_target[step]]

                # all_path_score
                if step == 0:
                    previous = score[step, START_ID, :]
                else:
                    # print(previous.shape)
                    previous = torch.logsumexp(
                        score[step, :, :] + previous.unsqueeze(1).expand(self.tagset_size, self.tagset_size),
                        dim=0
                    )
                            
            # 加上序列左右字符转移到<END>的概率
            gold_path_score += self.transition[current_target[-1], END_ID]
            all_path_score = torch.logsumexp(previous + self.transition[:, END_ID], dim=0)
            losses += all_path_score - gold_path_score

        batch_size = scores.shape[0]
        loss = losses / batch_size
        return loss

    def neg_log_likelihood(self, crf_scores, targets):
        """crf loss
        此方法并行计算batch中每个样本的loss 并最终求和
        此方法计算消耗时长较短

        crf loss 负对数似然损失
        loss        = exp(gold_path) / sum(exp(all_path))
        log(loss)   = log(exp(gold_path)) - log(sum(exp(all_path)))
                    = gold_path_score - all_path_score
        -log(loss)  = all_path_score = gold_path)score"""
        
        assert len(crf_scores) == len(targets)
        PAD_ID = self.tag2id['<pad>']
        START_ID = self.tag2id['<start>']
        END_ID = self.tag2id['<end>']
        device = targets.device
        mask = (targets != PAD_ID)
        batch_size, max_len = targets.shape
        lengths = mask.sum(dim=1) # 每一个序列的实际长度

        # gold_path_score 正确标签得分的和
        #     crf_scores矩阵形为[batch_size, max_len, tagset_size, tagset_size]
        #     crf_scores的第一个维度以句子为单位 crf_scores的第二个维度以单个字为单位
        #     每个字由一个[i=tagset_size, j=tagset_size]矩阵表示，该矩阵的具体含义为
        #     在前一个字的标签为i的前提下，当前字的标签为j的概率
        # 即 golden_socres 的值为每一个字矩阵中正确[i, j]下标的概率值之和
        
        # target           tagset_size*target_size的矩阵  保存该字正确的tag索引
        # former_target    tagset_size*target_size的矩阵  保存该字前一个字正确的tag索引
        former_targets = torch.zeros_like(targets)
        former_targets[:, 0] = START_ID
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

        # 抽取每个样本最后一个字符的tag索引
        last_tag_index = targets.gather(dim=1, index=(lengths.unsqueeze(-1)-1)).squeeze(1)
        # 计算每个样本最后一个字符转换到<end>标签的概率
        end_score = self.transition[:, END_ID].gather(dim=0, index=last_tag_index)

        # 所有样本目标路径去除 pad 求和 + 所有样本最后一个字符由正确tag转移到<end>标签的概率 求和
        gold_score = crf_score_i_j.masked_select(mask.unsqueeze(-1).unsqueeze(-1)).sum() + end_score.sum()



        # all_path_score 计算所有可能的值的和
        current_scores = torch.zeros(batch_size, self.tagset_size).to(device)
        for step in range(max_len):
            # 当前时刻 有效的batch_size（因为有些序列比较短)
            batch_size_step = (lengths > step).sum().item()
            if step == 0:
                current_scores[:batch_size_step] = crf_scores[:batch_size_step, step, START_ID, :]
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
        # 所有样本到达最后一个字符的所有可能路径继续转移到<end>标签的路径
        all_scores = torch.logsumexp(current_scores + self.transition[:, END_ID], dim=1).sum()

        loss = (all_scores - gold_score) / batch_size
        return loss

    def vibiter_decoding(self, crf_score, tag2id):
        """viterbi decoding"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']

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

        # 最后一个字符所属的所有标签中继续转移到<end>标签得分最高的tag，便是最后一个字符的tag
        best_end_idx = torch.argmax(max_scores + self.transition[:,end_id]).item()
        best_path = [best_end_idx]
        for step in range(seq_len-1, 0, -1):
            best_path.append(backpointer[step, best_path[-1]].item())
        best_path.reverse()

        return best_path