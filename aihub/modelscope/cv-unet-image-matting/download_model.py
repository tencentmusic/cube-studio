

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('portrait-matting', 'damo/cv_unet_image-matting')