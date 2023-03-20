

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('inverse-text-processing', 'damo/speech_inverse_text_processing_fun-text-processing-itn-ko')