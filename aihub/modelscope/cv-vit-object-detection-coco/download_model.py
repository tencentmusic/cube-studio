

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-object-detection', 'damo/cv_vit_object-detection_coco')