

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-super-resolution', 'damo/cv_realbasicvsr_video-super-resolution_videolq')