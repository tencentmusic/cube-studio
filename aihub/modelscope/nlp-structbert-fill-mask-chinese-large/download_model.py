

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('fill-mask', 'damo/nlp_structbert_fill-mask_chinese-large')