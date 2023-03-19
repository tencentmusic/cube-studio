

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-classification', 'damo/ofa_text-classification_mnli_large_en')