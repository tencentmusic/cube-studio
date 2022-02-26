
class K8SFailedException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class K8SJOBTimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
