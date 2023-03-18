

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('universal-matting', 'damo/cv_unet_universal-matting')