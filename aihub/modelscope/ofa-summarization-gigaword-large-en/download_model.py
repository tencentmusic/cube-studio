

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-summarization', 'damo/ofa_summarization_gigaword_large_en')