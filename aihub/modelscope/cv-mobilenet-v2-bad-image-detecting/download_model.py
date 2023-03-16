

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('bad-image-detecting', 'damo/cv_mobilenet-v2_bad-image-detecting')