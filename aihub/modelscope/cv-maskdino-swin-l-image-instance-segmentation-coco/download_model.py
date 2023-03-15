

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-segmentation', 'damo/cv_maskdino-swin-l_image-instance-segmentation_coco')