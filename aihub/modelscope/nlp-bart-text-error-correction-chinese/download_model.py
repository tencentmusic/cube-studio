

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-error-correction', 'damo/nlp_bart_text-error-correction_chinese')