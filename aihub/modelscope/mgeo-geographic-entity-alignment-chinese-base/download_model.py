

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('sentence-similarity', 'damo/mgeo_geographic_entity_alignment_chinese_base')