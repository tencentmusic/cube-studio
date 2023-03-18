

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-face-fusion', 'damo/cv_unet-image-face-fusion_damo')