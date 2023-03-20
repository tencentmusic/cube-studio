

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('auto-speech-recognition', 'damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch')