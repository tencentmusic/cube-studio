import config
import torch


def build_maps(lists) -> dict:
    maps = {}
    for e in lists:
        if e not in maps:
            maps[e] = len(maps)
    return maps


def readfile(filename):
    with open(config.data_path + filename, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
    return build_maps(data)


def token(word_lists, word2id, device):
    """
    :param word_lists: list of words
    :param word2id: dictionary of word2id
    """
    for idx in range(len(word_lists)):
        word_lists[idx].append('<end>')

    word_id_lists = (torch.ones(size=(len(word_lists), len(word_lists[0])), dtype=torch.long) * word2id[
        '<pad>']).to(device if torch.cuda.is_available() else 'cpu')
    for i in range(len(word_lists)):
        for j in range(len(word_lists[i])):
            word_id_lists[i][j] = word2id.get(word_lists[i][j],word2id['<unk>'])  # 遇到词表中不存在的字符，使用<unk>代替
        word_id_lists[i][-1] = word2id.get(word_lists[i][-1], word2id['<unk>'])
    return word_id_lists


# def token(word_lists, word2id, device):
#     """
#     :param word_lists: list of words
#     :param word2id: dictionary of word2id
#     """
#     word_lists.append('<end>')
#
#     word_id_lists = (torch.ones(size=len(word_lists), dtype=torch.long) * word2id[
#         '<pad>']).to(device if torch.cuda.is_available() else 'cpu')
#
#     for j in range(len(word_id_lists)):
#         word_id_lists[j] = word2id.get(word_lists[j],word2id['<unk>'])  # 遇到词表中不存在的字符，使用<unk>代替
#     word_id_lists[-1] = word2id.get(word_lists[-1], word2id['<unk>'])
#     return word_id_lists


def viterbi_decoding(crf_score, tag2id):
    """viterbi decoding
    不支持 batch"""
    start_id = tag2id['<start>']
    end_id = tag2id['<end>']

    device = crf_score.device
    seq_len = crf_score.shape[0]
    viterbi = torch.zeros(seq_len, len(tag2id)).to(device)
    backpointer = (torch.ones(size=(seq_len, len(tag2id)), dtype=torch.long) * end_id).to(device)

    for step in range(seq_len):
        if step == 0:  # 第一个字
            viterbi[step, :] = crf_score[step, start_id, :]
            backpointer[step, :] = start_id
        else:
            max_scores, prev_tags_id = torch.max(
                viterbi[step - 1, :].unsqueeze(1) + crf_score[step, :, :],
                dim=0
            )
            viterbi[step, :] = max_scores
            backpointer[step, :] = prev_tags_id

    best_end_idx = end_id
    best_path = []
    for step in range(seq_len - 1, 0, -1):
        if step == seq_len - 1:
            best_path.append(backpointer[step, best_end_idx].item())
        else:
            best_path.append(backpointer[step, best_path[-1]].item())
    best_path.reverse()

    return best_path


def predtion_to_tags(prediction, id2tag):
    return [id2tag.get(id, 'O') for id in prediction]


def idlist2tag(lists, tag2id, id2tag):
    pred_tag_lists = []
    best_path = viterbi_decoding(lists[0], tag2id)
    pred_tag_lists.append(
        predtion_to_tags(
            best_path, id2tag
        )
    )
    return pred_tag_lists
