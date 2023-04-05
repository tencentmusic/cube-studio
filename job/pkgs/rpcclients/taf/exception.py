# -*- encoding=utf-8 -*-
class TafException(Exception):
    pass


class TafJceDecodeRequireNotExist(TafException):
    pass


class TafJceDecodeMismatch(TafException):
    pass


class TafJceDecodeInvalidValue(TafException):
    pass


class TafJceUnsupportType(TafException):
    pass


class TafNetConnectException(TafException):
    pass


class TafNetConnectLostException(TafException):
    pass


class TafNetSocketException(TafException):
    pass


class TafProxyDecodeException(TafException):
    pass


class TafProxyEncodeException(TafException):
    pass


class TafServerEncodeException(TafException):
    pass


class TafServerDecodeException(TafException):
    pass


class TafServerNoFuncException(TafException):
    pass


class TafServerNoServantException(TafException):
    pass


class TafServerQueueTimeoutException(TafException):
    pass


class TafServerUnknownException(TafException):
    pass


class TafSyncCallTimeoutException(TafException):
    pass


class TafRegistryException(TafException):
    pass


class TafServerResetGridException(TafException):
    pass
