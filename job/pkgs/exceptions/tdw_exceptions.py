# -*- coding: utf-8 -*-


class TDWNoResException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TDWFailedException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)