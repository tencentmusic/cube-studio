

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-to-speech', 'speech_tts/speech_sambert-hifigan_tts_chuangirl_Sichuan_16k')