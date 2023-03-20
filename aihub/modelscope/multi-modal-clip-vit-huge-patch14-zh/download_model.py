

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('multi-modal-embedding', 'damo/multi-modal_clip-vit-huge-patch14_zh')