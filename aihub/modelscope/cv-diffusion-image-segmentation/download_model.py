

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('semantic-segmentation', 'damo/cv_diffusion_image-segmentation')