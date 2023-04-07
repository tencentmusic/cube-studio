

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('document-segmentation', 'damo/nlp_bert_document-segmentation_chinese-base')