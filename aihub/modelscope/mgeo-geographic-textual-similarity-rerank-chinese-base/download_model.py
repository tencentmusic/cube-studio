

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-ranking', 'damo/mgeo_geographic_textual_similarity_rerank_chinese_base')