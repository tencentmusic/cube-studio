

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('keyword-spotting', 'damo/speech_dfsmn_kws_char_farfield_16k_nihaomiya')