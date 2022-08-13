from sklearn_crfsuite import CRF


class CRFModel:

    def __init__(
        self, 
        algorithm='lbfgs', 
        c1=0.1, 
        c2=0.1, 
        max_iterations=100, 
        all_possible_transitions=False):
        self.model = CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_states=all_possible_transitions)

    def train(self, sentences, tag_lists):
        """训练模型

        :param sentences --> train_word_lists is a list of lists of sentence
        :param tag_lists --> train_tag_lists
        """
        features = [sent2features(sent) for sent in sentences]
        self.model.fit(features, tag_lists)

    def test(self,sentences):
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists

def word2features(sent, i):
    """抽取单个字的特征"""
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i-1]
    next_word = "</s>" if i == (len(sent) - 1) else sent[i+1]
    # 使用的特征：
    # 前一个词，当前词，后一个词
    # 前一个词+当前词，当前词+后一个词
    feature = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word + word,
        'w:w+1': word + next_word,
        'bias': 1
    }
    return feature

def sent2features(sent):
    """抽取序列特征"""
    return [word2features(sent, i) for i in range(len(sent))]