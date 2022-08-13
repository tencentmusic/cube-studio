from models.HMM import HMM
from models.CRF import CRFModel
from bilstm_opration import BiLSTM_opration
from bilstm_crf_opration import BiLSTM_CRF_opration
from evaluating import Metrics
from utils.utils import save_model, add_end_tag


def hmm_train_eval(train_data, test_data, word2id, tag2id, remove_0=False):
    """hmm模型的评估与训练"""
    print("hmm模型的评估与训练...")
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    # 模型训练
    hmm_model = HMM(N=len(tag2id), M=len(word2id))
    hmm_model.train(train_word_lists, train_tag_lists, word2id, tag2id)
    # save_model(hmm_model,"./ckpts/hmm.pkl")

    # 模型评估
    pred_tag_lists = hmm_model.test(test_word_lists, word2id, tag2id)
    metrics = Metrics(test_tag_lists, pred_tag_lists)
    metrics.report_scores(dtype='HMM')

    return hmm_model


def crf_train_eval(train_data, test_data, remove_0=False):
    """crf模型的评估与训练"""
    print("crf模型的评估与训练")
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data
    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    # save_model(crf_model, "./ckpts/crf.pkl")

    pred_tag_lists = crf_model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, pred_tag_lists)
    metrics.report_scores(dtype='CRF')

    return crf_model


def bilstm_train_eval(train_data, dev_data, test_data, word2id, tag2id):
    """BiLSTM模型的评估与训练"""
    print("BiLSTM模型的评估与训练")

    bilstm_model = BiLSTM_opration(
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data,
        word2id=word2id,
        tag2id=tag2id
    )
    bilstm_model.train()
    bilstm_model.evaluate()

    return bilstm_model


def bilstm_crf_train_eval(train_data, dev_data, test_data, word2id, tag2id):
    """BiLSTM_CRF模型的评估与训练"""
    print("BiLSTM_CRF模型的评估与训练")

    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    train_word_lists, train_tag_lists = add_end_tag(train_word_lists, train_tag_lists)
    dev_word_lists, dev_tag_lists = add_end_tag(dev_word_lists, dev_tag_lists)
    test_word_lists = add_end_tag(test_word_lists)

    bilstm_crf_model = BiLSTM_CRF_opration(
        train_data=(train_word_lists, train_tag_lists),
        dev_data=(dev_word_lists, dev_tag_lists),
        test_data=(test_word_lists, test_tag_lists),
        word2id=word2id,
        tag2id=tag2id
    )

    bilstm_crf_model.train()
    bilstm_crf_model.evaluate()

    return bilstm_crf_model
