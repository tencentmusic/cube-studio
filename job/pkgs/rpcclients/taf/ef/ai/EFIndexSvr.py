from job.pkgs.rpcclients.taf.core import tafcore
from job.pkgs.rpcclients.taf.__rpc import ServantProxy
from .ai_hello import *

import typing


class E_EF_UPDATE_TYPE:
    pass


class STUpdateEfIndexReq(tafcore.struct):
    __taf_class__ = "ai.STUpdateEfIndexReq"

    def __init__(self):
        self.project = ""
        self.model_name = ""
        self.version = ""
        self.type = ""
        self.index_id = 0
        self.date = ""

    @staticmethod
    def writeTo(oos, value):
        oos.write(tafcore.string, 0, value.project)
        oos.write(tafcore.string, 1, value.model_name)
        oos.write(tafcore.string, 2, value.version)
        oos.write(tafcore.string, 3, value.type)
        oos.write(tafcore.int32, 4, value.index_id)
        oos.write(tafcore.string, 5, value.date)

    @staticmethod
    def readFrom(ios):
        value = STUpdateEfIndexReq()
        value.project = ios.read(tafcore.string, 0, False, value.project)
        value.model_name = ios.read(tafcore.string, 1, False, value.model_name)
        value.version = ios.read(tafcore.string, 2, False, value.version)
        value.type = ios.read(tafcore.string, 3, False, value.type)
        value.index_id = ios.read(tafcore.int32, 4, False, value.index_id)
        value.date = ios.read(tafcore.string, 5, False, value.date)
        return value


class STUpdateEfIndexRsp(tafcore.struct):
    __taf_class__ = "ai.STUpdateEfIndexRsp"

    def __init__(self):
        self.taskid = ""

    @staticmethod
    def writeTo(oos, value):
        oos.write(tafcore.string, 0, value.taskid)

    @staticmethod
    def readFrom(ios):
        value = STUpdateEfIndexRsp()
        value.taskid = ios.read(tafcore.string, 0, False, value.taskid)
        return value


class STGetEmbStatsReq(tafcore.struct):
    __taf_class__ = "ai.STGetEmbStatsReq"

    def __init__(self):
        self.project = ""
        self.model_name = ""
        self.version = ""

    @staticmethod
    def writeTo(oos, value):
        oos.write(tafcore.string, 0, value.project)
        oos.write(tafcore.string, 1, value.model_name)
        oos.write(tafcore.string, 2, value.version)

    @staticmethod
    def readFrom(ios):
        value = STGetEmbStatsReq()
        value.project = ios.read(tafcore.string, 0, True, value.project)
        value.model_name = ios.read(tafcore.string, 1, True, value.model_name)
        value.version = ios.read(tafcore.string, 2, True, value.version)
        return value


class STGetEmbStatsRsp(tafcore.struct):
    __taf_class__ = "ai.STGetEmbStatsRsp"
    mapcls_data = tafcore.mapclass(tafcore.string, tafcore.string)

    def __init__(self):
        self.code = 0
        self.msg = ""
        self.data = STGetEmbStatsRsp.mapcls_data()

    @staticmethod
    def writeTo(oos, value):
        oos.write(tafcore.int32, 0, value.code)
        oos.write(tafcore.string, 1, value.msg)
        oos.write(value.mapcls_data, 2, value.data)

    @staticmethod
    def readFrom(ios):
        value = STGetEmbStatsRsp()
        value.code = ios.read(tafcore.int32, 0, False, value.code)
        value.msg = ios.read(tafcore.string, 1, False, value.msg)
        value.data = ios.read(value.mapcls_data, 2, False, value.data)
        return value


class STRemoveModelEmbVersionReq(tafcore.struct):
    __taf_class__ = "ai.STRemoveModelEmbVersionReq"

    def __init__(self):
        self.project = ""
        self.model_name = ""
        self.version = ""

    @staticmethod
    def writeTo(oos, value):
        oos.write(tafcore.string, 0, value.project)
        oos.write(tafcore.string, 1, value.model_name)
        oos.write(tafcore.string, 2, value.version)

    @staticmethod
    def readFrom(ios):
        value = STRemoveModelEmbVersionReq()
        value.project = ios.read(tafcore.string, 0, True, value.project)
        value.model_name = ios.read(tafcore.string, 1, True, value.model_name)
        value.version = ios.read(tafcore.string, 2, True, value.version)
        return value


class STRemoveModelEmbVersionRsp(tafcore.struct):
    __taf_class__ = "ai.STRemoveModelEmbVersionRsp"

    def __init__(self):
        self.success = True

    @staticmethod
    def writeTo(oos, value):
        oos.write(tafcore.boolean, 0, value.success)

    @staticmethod
    def readFrom(ios):
        value = STRemoveModelEmbVersionRsp()
        value.success = ios.read(tafcore.boolean, 0, False, value.success)
        return value


# proxy for client
class EFIndexSvrProxy(ServantProxy):
    def hello(self, req, context=ServantProxy.mapcls_context()) -> typing.Tuple[int, SHelloRsp]:
        oos = tafcore.JceOutputStream()
        oos.write(SHelloReq, 1, req)

        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "hello", oos.getBuffer(), context, None)

        ios = tafcore.JceInputStream(rsp.sBuffer)
        ret = ios.read(tafcore.int32, 0, True)
        rsp = ios.read(SHelloRsp, 2, True)

        return ret, rsp

    def updateIndex(self, req, context=ServantProxy.mapcls_context()) -> typing.Tuple[int, STUpdateEfIndexRsp]:
        oos = tafcore.JceOutputStream()
        oos.write(STUpdateEfIndexReq, 1, req)

        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "updateIndex", oos.getBuffer(), context, None)

        ios = tafcore.JceInputStream(rsp.sBuffer)
        ret = ios.read(tafcore.int32, 0, True)
        rsp = ios.read(STUpdateEfIndexRsp, 2, True)

        return ret, rsp

    def checkEmbStats(self, req, context=ServantProxy.mapcls_context()) -> typing.Tuple[int, STGetEmbStatsRsp]:
        oos = tafcore.JceOutputStream()
        oos.write(STGetEmbStatsReq, 1, req)

        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "checkEmbStats", oos.getBuffer(), context, None)

        ios = tafcore.JceInputStream(rsp.sBuffer)
        ret = ios.read(tafcore.int32, 0, True)
        rsp = ios.read(STGetEmbStatsRsp, 2, True)

        return ret, rsp

    def removeModelEmb(self, req, context=ServantProxy.mapcls_context()) \
            -> typing.Tuple[int, STRemoveModelEmbVersionRsp]:
        oos = tafcore.JceOutputStream()
        oos.write(STRemoveModelEmbVersionReq, 1, req)

        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "removeModelEmb", oos.getBuffer(), context, None)

        ios = tafcore.JceInputStream(rsp.sBuffer)
        ret = ios.read(tafcore.int32, 0, True)
        rsp = ios.read(STRemoveModelEmbVersionRsp, 2, True)

        return ret, rsp
