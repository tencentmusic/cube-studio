

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-classification', 'damo/nlp_user-satisfaction-estimation_chinese')