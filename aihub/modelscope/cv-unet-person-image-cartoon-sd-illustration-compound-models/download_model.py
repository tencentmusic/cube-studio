

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-portrait-stylization', 'damo/cv_unet_person-image-cartoon-sd-illustration_compound-models')