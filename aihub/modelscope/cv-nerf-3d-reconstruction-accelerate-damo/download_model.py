

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('nerf-recon-acc', 'damo/cv_nerf-3d-reconstruction-accelerate_damo')