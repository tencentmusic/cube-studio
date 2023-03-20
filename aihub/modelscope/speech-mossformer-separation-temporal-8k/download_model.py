

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('speech-separation', 'damo/speech_mossformer_separation_temporal_8k')