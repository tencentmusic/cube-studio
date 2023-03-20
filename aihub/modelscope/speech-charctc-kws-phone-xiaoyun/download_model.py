

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('keyword-spotting', 'damo/speech_charctc_kws_phone-xiaoyun')