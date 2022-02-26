
import re
import os
from datetime import datetime, timedelta

from .constants import PipelineParam


def flatten_seq(sequence):
    if not sequence:
        return sequence

    for item in sequence:
        if isinstance(item, (tuple, list)):
            for subitem in flatten_seq(item):
                yield subitem
        else:
            yield item
