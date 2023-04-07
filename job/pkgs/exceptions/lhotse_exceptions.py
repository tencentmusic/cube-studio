# coding=utf-8
# @Time     : 2021/4/16 15:01
# @Auther   : lionpeng@tencent.com


class CheckTaskException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TaskAbnormalException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QueryTaskException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QueryTaskInstanceException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TaskInstanceFailedException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class WaitTimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)