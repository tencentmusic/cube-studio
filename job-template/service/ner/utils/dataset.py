from inspect import classify_class_attrs
from numpy import dtype
from torch.utils.data import Dataset
import torch

class WordTagDataset(Dataset):

    def __init__(self, word_lists, tag_lists, vocabulary, tag2id) -> None:
        super(WordTagDataset).__init__()
        assert len(word_lists) == len(tag_lists)

        # pairs = list(zip(word_lists,tag_lists))
        # indices = sorted(range(len(pairs)),key=lambda x:len(pairs[x][0]),reverse=True)
        # pairs = [pairs[i] for i in indices]
        # self.word_lists, self.tag_lists = list(zip(*pairs))

        self.word_lists = word_lists
        self.tag_lists = tag_lists
        self.vocabulary = vocabulary
        self.tag2id = tag2id

    def __getitem__(self, index):
        wordID_list = [self.vocabulary.get(word, self.vocabulary['<unk>']) for word in self.word_lists[index]]
        tagID_list = [self.tag2id.get(tag, self.tag2id['<unk>']) for tag in self.tag_lists[index]]
        MAX_PADDING = 64
        seq_len = len(wordID_list)
        if seq_len < MAX_PADDING:
            for i in range(MAX_PADDING - seq_len):
                wordID_list.append(self.vocabulary['<pad>'])
                tagID_list.append(self.tag2id['<pad>'])
        else:
            wordID_list = wordID_list[0:MAX_PADDING]
            tagID_list = tagID_list[0:MAX_PADDING]
        # print(torch.tensor(wordID_list, dtype=torch.long))
        # print(torch.tensor(tagID_list, dtype=torch.long))

        return torch.tensor(wordID_list, dtype=torch.long), torch.tensor(tagID_list, dtype=torch.long)


    def __len__(self):
        return len(self.word_lists)


                
