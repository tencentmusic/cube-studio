

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-detection', 'damo/cv_ddsar_face-detection_iclr23-damofd-34G')