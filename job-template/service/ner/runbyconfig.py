import bentoml
import config
from data import build_corpus
from minio import Minio
import os
from train_evaluate import hmm_train_eval, crf_train_eval, bilstm_train_eval, bilstm_crf_train_eval
from utils.preprocessing import Preprocessing, save2id
from utils.utils import save_model
from config import TrainingConfig, BiLSTMCRFTrainConfig
from loguru import logger



if __name__ == "__main__":
    if not os.path.exists(config.data_path+config.data_name):
        # contact to Minio
        minio_client = Minio(
            '10.101.32.11:9000',
            access_key='admin',
            secret_key='root123456',
            secure=False
        )
        try:
            minio_client.fget_object(
                bucket_name='data',
                object_name='people_daily_BIO.txt',
                file_path=config.data_path + config.data_name
            )
        except BaseException as err:
            print(err)
    # preprocessing data
    data_preprocessing = Preprocessing(
        file_path=config.data_path,
        file_name=config.data_name
    )
    data_preprocessing.train_test_dev_split(data_rate=config.data_rate)
    data_preprocessing.construct_vocabulary_labels()

    # load data
    logger.info('long data ...')
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train", data_dir=config.data_path)
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False, data_dir=config.data_path)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir=config.data_path)
    model = None
    if config.model_name == 'HMM':
        # train and evaluate HMM model
        logger.info("HMM_train_eval start")
        model = hmm_train_eval(
            train_data=(train_word_lists, train_tag_lists),
            test_data=(test_word_lists, test_tag_lists),
            word2id=word2id,
            tag2id=tag2id
        )
        save2id(word2id, tag2id, config.model_name)
        save_model(model, './ckpts/{}model.pkl'.format(config.model_name))
        saved_model = bentoml.picklable_model.save_model(
            config.model_name,
            model,
            signatures={"__call__": {"batchable": True}}
        )
        print(f"Model saved: {saved_model}")
        logger.info("HMM_train_eval end")

    elif config.model_name == 'CRF':
        # train and evaluate CRF model
        model = crf_train_eval(
            train_data=(train_word_lists, train_tag_lists),
            test_data=(test_word_lists, test_tag_lists)
        )
    elif config.model_name == 'BiLSTM':
        # BiLSTM
        TrainingConfig.batch_size = config.batch_size
        TrainingConfig.epochs = config.epochs
        TrainingConfig.lr = config.lr
        model = bilstm_train_eval(
            train_data=(train_word_lists, train_tag_lists),
            dev_data=(dev_word_lists, dev_tag_lists),
            test_data=(test_word_lists, test_tag_lists),
            word2id=word2id,
            tag2id=tag2id
        )
        save2id(word2id, tag2id, config.model_name)
        bentoml.pytorch.save_model('BiLSTM', model.model)
        save_model(model, './ckpts/{}model.pkl'.format(config.model_name))
    elif config.model_name == 'BiLSTM_CRF':
        # BiLSTM CRF
        BiLSTMCRFTrainConfig.batch_size = config.batch_size
        BiLSTMCRFTrainConfig.epochs = config.epochs
        BiLSTMCRFTrainConfig.lr = config.lr
        model = bilstm_crf_train_eval(
            train_data=(train_word_lists, train_tag_lists),
            dev_data=(dev_word_lists, dev_tag_lists),
            test_data=(test_word_lists, test_tag_lists),
            word2id=word2id,
            tag2id=tag2id
        )
        save2id(word2id, tag2id, config.model_name)
        bentoml.pytorch.save_model('BiLSTM_CRF', model.model)
        save_model(model, './ckpts/{}model.pkl'.format(config.model_name))

