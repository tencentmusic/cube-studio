

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-classification', 'damo/cv_convnext-base_image-classification_garbage')