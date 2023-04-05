

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('sentence-similarity', 'damo/nlp_translation_quality_estimation_multilingual')