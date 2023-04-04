# coding=utf-8
# @Time     : 2021/4/12 15:58
# @Auther   : lionpeng@tencent.com


class StartFLowException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QueryFlowException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FlowFailedException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FLowTimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
