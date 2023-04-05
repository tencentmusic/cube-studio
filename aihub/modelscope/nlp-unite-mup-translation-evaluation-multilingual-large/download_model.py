

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('translation-evaluation', 'damo/nlp_unite_mup_translation_evaluation_multilingual_large')