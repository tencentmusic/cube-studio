

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-object-detection', 'damo/cv_swinl_image-object-detection_dino')