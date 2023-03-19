

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('auto-speech-recognition', 'damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch')