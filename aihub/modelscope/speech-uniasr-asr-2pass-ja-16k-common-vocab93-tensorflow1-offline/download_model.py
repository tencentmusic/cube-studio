

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('auto-speech-recognition', 'damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline')