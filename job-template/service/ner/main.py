from data import build_corpus
from train_evaluate import hmm_train_eval, crf_train_eval, bilstm_train_eval


def main():

    # load data
    print('long data ...')
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # train and evaluate HMM model
    hmm_pred = hmm_train_eval(
        train_data=(train_word_lists, train_tag_lists),
        test_data=(test_word_lists, test_tag_lists),
        word2id=word2id,
        tag2id=tag2id
    )

    # train and evaluate CRF model
    crf_pred = crf_train_eval(
        train_data=(train_word_lists, train_tag_lists),
        test_data=(test_word_lists, test_tag_lists)
    )

    # BiLSTM
    bilstm_pred = bilstm_train_eval(
        train_data=(train_word_lists, train_tag_lists),
        dev_data=(dev_word_lists, dev_tag_lists),
        test_data=(test_word_lists, test_tag_lists),
        word2id=word2id,
        tag2id=tag2id
    )
    

main()