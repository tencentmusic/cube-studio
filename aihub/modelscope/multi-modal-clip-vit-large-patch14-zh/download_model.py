

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('multi-modal-embedding', 'damo/multi-modal_clip-vit-large-patch14_zh')