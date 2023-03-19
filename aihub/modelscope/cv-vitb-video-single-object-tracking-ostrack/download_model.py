

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-single-object-tracking', 'damo/cv_vitb_video-single-object-tracking_ostrack')