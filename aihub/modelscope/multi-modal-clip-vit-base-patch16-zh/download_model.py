

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('multi-modal-embedding', 'damo/multi-modal_clip-vit-base-patch16_zh')