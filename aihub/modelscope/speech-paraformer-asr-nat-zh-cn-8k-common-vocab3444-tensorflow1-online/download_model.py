

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('auto-speech-recognition', 'damo/speech_paraformer_asr_nat-zh-cn-8k-common-vocab3444-tensorflow1-online')