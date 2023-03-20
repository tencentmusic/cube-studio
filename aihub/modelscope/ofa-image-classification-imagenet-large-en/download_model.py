

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-classification', 'damo/ofa_image-classification_imagenet_large_en')