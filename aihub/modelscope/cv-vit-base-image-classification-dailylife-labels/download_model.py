

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-classification', 'damo/cv_vit-base_image-classification_Dailylife-labels')