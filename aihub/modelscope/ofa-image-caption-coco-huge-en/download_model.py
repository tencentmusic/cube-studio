

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-captioning', 'damo/ofa_image-caption_coco_huge_en')