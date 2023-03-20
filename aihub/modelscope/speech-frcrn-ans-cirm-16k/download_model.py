

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('acoustic-noise-suppression', 'damo/speech_frcrn_ans_cirm_16k')