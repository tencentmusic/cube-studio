

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-driving-perception', 'damo/cv_yolopv2_image-driving-perception_bdd100k')