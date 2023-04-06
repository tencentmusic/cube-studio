from job.pkgs.rpcclients.taf.core import tafcore
from job.pkgs.rpcclients.taf.__rpc import ServantProxy


class SHelloReq(tafcore.struct):
    __taf_class__ = "ai.SHelloReq"

    def __init__(self):
        self.uin = 0

    @staticmethod
    def writeTo(oos, value):
        oos.write(tafcore.int64, 0, value.uin)

    @staticmethod
    def readFrom(ios):
        value = SHelloReq()
        value.uin = ios.read(tafcore.int64, 0, False, value.uin)
        return value


class SHelloRsp(tafcore.struct):
    __taf_class__ = "ai.SHelloRsp"

    def __init__(self):
        self.msg = ""

    @staticmethod
    def writeTo(oos, value):
        oos.write(tafcore.string, 0, value.msg)

    @staticmethod
    def readFrom(ios):
        value = SHelloRsp()
        value.msg = ios.read(tafcore.string, 0, False, value.msg)
        return value
