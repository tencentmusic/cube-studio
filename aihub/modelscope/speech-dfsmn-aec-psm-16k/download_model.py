

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('acoustic-echo-cancellation', 'damo/speech_dfsmn_aec_psm_16k')