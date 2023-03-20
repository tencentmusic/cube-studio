

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('language-score-prediction', 'damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch')