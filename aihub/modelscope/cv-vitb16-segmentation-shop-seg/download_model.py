

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('shop-segmentation', 'damo/cv_vitb16_segmentation_shop-seg')