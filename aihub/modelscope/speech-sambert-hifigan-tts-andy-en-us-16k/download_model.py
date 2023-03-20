

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-to-speech', 'damo/speech_sambert-hifigan_tts_andy_en-us_16k')