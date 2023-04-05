

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('faq-question-answering', 'damo/nlp_structbert_faq-question-answering_chinese-finance-base')