

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('auto-speech-recognition', 'damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404')