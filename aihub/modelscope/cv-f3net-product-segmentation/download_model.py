

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('product-segmentation', 'damo/cv_F3Net_product-segmentation')