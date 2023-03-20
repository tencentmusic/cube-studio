

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-captioning', 'damo/mplug_image-captioning_coco_large_en')