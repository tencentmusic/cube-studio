

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-to-image-synthesis', 'damo/cv_diffusion_text-to-image-synthesis_tiny')