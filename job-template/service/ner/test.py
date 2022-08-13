import bentoml
import torch
import config
import pickle
from utils.bilstm_crf_token import readfile, token, idlist2tag

word2id = readfile(config.model_name+'_voc.txt')
tag2id = readfile(config.model_name+'_tags.txt')
id2tag = dict((ids, tag) for tag, ids in tag2id.items())
device = config.device
model = bentoml.pytorch.load_model("bilstm_crf:latest")
# model = bentoml.pytorch.load_model("bilstm:latest")

sentence = '1962年1月出生，南京工学院毕业'

# with open('./ckpts/BiLSTM_CRFmodel.pkl', 'rb') as f:
#     modela = pickle.load(f)
# tags = modela.predict_sentence(sentence)
# print(tags)



word_list = []
for i in sentence:
    word_list.append(i)
word_list = ['1', '9', '6', '2', '年', '1', '月', '出', '生', '，', '南', '京', '工', '学', '院', '毕', '业', '。']
inputs = token([word_list], word2id, device)
# output = model.forward(inputs)
# tags = idlist2tag(output, tag2id, id2tag)
# print(tags)

runner = bentoml.pytorch.get('bilstm_crf:latest').to_runner()
runner.init_local()
outputs = runner.__call__.run(inputs)
tags = idlist2tag(outputs, tag2id, id2tag)
print(tags)
