

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-matching', 'damo/cv_quadtree_attention_image-matching_outdoor')