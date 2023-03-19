

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-text-retrieval', 'damo/cv_vit-b32_retrieval_vop_bias')