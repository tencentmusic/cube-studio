

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('fill-mask', 'damo/nlp_bert_fill-mask_chinese-base')