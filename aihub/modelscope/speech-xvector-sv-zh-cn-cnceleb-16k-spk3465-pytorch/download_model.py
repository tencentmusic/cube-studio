

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('speaker-verification', 'damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch')