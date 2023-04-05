

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('table-question-answering', 'damo/nlp_star_conversational-text-to-sql')