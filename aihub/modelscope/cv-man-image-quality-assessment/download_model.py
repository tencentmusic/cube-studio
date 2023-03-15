

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-quality-assessment-mos', 'damo/cv_man_image-quality-assessment')