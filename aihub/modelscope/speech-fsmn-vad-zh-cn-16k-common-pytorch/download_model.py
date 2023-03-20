

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('voice-activity-detection', 'damo/speech_fsmn_vad_zh-cn-16k-common-pytorch')