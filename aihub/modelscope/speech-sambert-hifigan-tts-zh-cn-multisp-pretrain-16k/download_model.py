

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-to-speech', 'speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_16k')