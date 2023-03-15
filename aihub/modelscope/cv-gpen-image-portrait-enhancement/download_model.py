

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-portrait-enhancement', 'damo/cv_gpen_image-portrait-enhancement')