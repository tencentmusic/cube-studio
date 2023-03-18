

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-inpainting', 'damo/cv_stable-diffusion-v2_image-inpainting_base')