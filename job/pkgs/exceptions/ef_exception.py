# coding=utf-8
# @Time     : 2020/10/28 15:35
# @Auther   : lionpeng@tencent.com


class StartDeployEmbeddingError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QueryEmbeddingDeployError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
