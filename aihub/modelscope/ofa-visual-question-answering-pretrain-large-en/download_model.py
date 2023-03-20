

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('visual-question-answering', 'damo/ofa_visual-question-answering_pretrain_large_en')