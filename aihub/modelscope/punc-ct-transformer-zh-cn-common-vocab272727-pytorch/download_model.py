

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('punctuation', 'damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch')