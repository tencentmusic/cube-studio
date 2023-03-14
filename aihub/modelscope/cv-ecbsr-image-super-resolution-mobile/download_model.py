

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-super-resolution', 'damo/cv_ecbsr_image-super-resolution_mobile')