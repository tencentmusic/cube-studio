

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('document-segmentation', 'damo/nlp_ponet_document-segmentation_topic-level_chinese-base')