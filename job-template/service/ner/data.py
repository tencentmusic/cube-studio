import os


def build_corpus(split, make_vocab=True, data_dir='./zdata/'):
    """数据读取
    """

    assert split.lower() in ["train", "dev", "test"]

    word_lists = []
    tag_lists = []
    with open(os.path.join(data_dir, split + '.txt'), 'r') as f:
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


def token_to_str(word_lists):
    s = ' '
    sentence_list = []
    for word_list in word_lists:
        sentence_list.append(s.join(word_list))
    return sentence_list
