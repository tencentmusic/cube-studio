

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('named-entity-recognition', 'damo/nlp_structbert_keyphrase-extraction_base-icassp2023-mug-track4-baseline')