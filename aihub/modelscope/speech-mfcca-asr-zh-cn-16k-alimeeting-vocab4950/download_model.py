

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('auto-speech-recognition', 'NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950')