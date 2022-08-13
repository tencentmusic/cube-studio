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


def token(word_lists, word2id, device='cpu'):
    word_id_lists = (
            torch.ones(size=(len(word_lists), len(word_lists[0])), dtype=torch.long) * word2id['<pad>']).to(
        device if torch.cuda.is_available() else 'cpu')

    for i in range(len(word_lists)):
        for j in range(len(word_lists[i])):
            word_id_lists[i][j] = word2id.get(word_lists[i][j], word2id['<unk>'])  # 遇到词表中不存在的字符，使用<unk>代替

    return word_id_lists


def idlist2tag(prediction, tag2id, id2tag):
    pred_tag_lists = []
    pred_tag_lists.append([id2tag[ids.item()] for ids in torch.argmax(prediction, dim=2)[0]]
                          )
    return pred_tag_lists
