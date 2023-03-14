

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-color-enhancement', 'damo/cv_adaint_image-color-enhance-models')