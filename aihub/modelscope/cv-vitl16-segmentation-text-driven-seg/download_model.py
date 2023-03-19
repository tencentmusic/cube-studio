

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-driven-segmentation', 'damo/cv_vitl16_segmentation_text-driven-seg')