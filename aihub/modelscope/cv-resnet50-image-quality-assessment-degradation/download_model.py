

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-quality-assessment-degradation', 'damo/cv_resnet50_image-quality-assessment_degradation')