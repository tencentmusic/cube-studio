

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('auto-speech-recognition', 'damo/speech_UniASR_asr_2pass-id-16k-common-vocab1067-tensorflow1-online')