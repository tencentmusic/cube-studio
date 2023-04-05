

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('table-question-answering', 'damo/nlp_convai_text2sql_pretrain_cn')