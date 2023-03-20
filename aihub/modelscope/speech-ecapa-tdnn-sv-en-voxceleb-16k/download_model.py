

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('speaker-verification', 'damo/speech_ecapa-tdnn_sv_en_voxceleb_16k')