

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-generation', 'damo/spring_couplet_generation')