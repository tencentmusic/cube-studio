

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-text-retrieval', 'damo/mplug_image-text-retrieval_flickr30k_large_en')