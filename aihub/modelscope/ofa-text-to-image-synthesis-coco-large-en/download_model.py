

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-to-image-synthesis', 'damo/ofa_text-to-image-synthesis_coco_large_en')