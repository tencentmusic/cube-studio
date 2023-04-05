

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-ranking', 'damo/nlp_rom_passage-ranking_chinese-base')