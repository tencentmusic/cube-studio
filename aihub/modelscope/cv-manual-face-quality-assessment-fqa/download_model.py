

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-quality-assessment', 'damo/cv_manual_face-quality-assessment_fqa')