

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('auto-speech-recognition', 'wenet/u2pp_conformer-asr-cn-16k-online')