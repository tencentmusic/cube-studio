

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-to-speech', 'damo/speech_sambert-hifigan_tts_zhitian_emo_zh-cn_16k')