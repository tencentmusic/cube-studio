

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-classification', 'damo/cv_tinynas_classification')