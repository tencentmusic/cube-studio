

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('skin-retouching', 'damo/cv_unet_skin-retouching')