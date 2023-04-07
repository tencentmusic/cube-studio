

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('translation', 'damo/nlp_csanmt_translation_en2zh_base')