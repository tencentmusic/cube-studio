

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-multi-modal-embedding', 'damo/multi_modal_clip_vtretrival_msrvtt_53')