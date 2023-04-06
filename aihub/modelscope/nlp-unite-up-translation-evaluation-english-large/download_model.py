

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('translation-evaluation', 'damo/nlp_unite_up_translation_evaluation_English_large')