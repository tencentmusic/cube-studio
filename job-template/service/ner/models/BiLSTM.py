import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, dropout=0.1, batch_first=True):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=self.vocab_size - 1)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=batch_first
        )
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size, bias=True)

    def _init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        # self.hidden = self._init_hidden()
        self.word_embeds.weight.requires_grad = False
        sentence = sentence.long()
        embeds = self.word_embeds(sentence)
        lstm_out, _ = self.bilstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence):
        return self._get_lstm_features(sentence)


def cal_loss(prediction, targets, tag2id):
    """损失计算
    :param prediction [Batch_size, Seq_length, tagset_size]
    :param targets [Batch_size, Seq_length]
    """

    # <pad> 占位符 不参与loss计算
    PAD = tag2id.get('<pad>')
    pad_mask = (targets != PAD)

    targets = targets[pad_mask]
    prediction = prediction[pad_mask]

    # 交叉熵 loss
    loss = F.cross_entropy(prediction, targets)

    return loss
