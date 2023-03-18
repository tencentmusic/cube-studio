

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('semantic-segmentation', 'damo/cv_u2net_salient-detection')