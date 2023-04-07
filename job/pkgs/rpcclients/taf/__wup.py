# -*- encoding=utf-8 -*-
__author__ = "kevintian@tencnet.com (Kevin.Tian)"
__author__ = "kevintian@wuhan.tencent.com (Kevin.Tian)"

import struct
import string
import sys
from .__util import util
from .__jce import JceOutputStream
from .__jce import JceInputStream
from .__packet import RequestPacket


class TafUniPacket(object):
    def __init__(self):
        self.__mapa = util.mapclass(util.string, util.bytes)
        self.__mapv = util.mapclass(util.string, self.__mapa)
        self.__buffer = self.__mapv()
        self.__code = RequestPacket()

    @property
    def servant(self):
        return self.__code.sServantName

    @servant.setter
    def servant(self, value):
        self.__code.sServantName = value

    @property
    def func(self):
        return self.__code.sFuncName

    @func.setter
    def func(self, value):
        self.__code.sFuncName = value

    @property
    def requestid(self):
        return self.__code.iRequestId

    @requestid.setter
    def requestid(self, value):
        self.__code.iRequestId = value

    @property
    def result_code(self):
        if not self.__code.status.has_key("STATUS_RESULT_CODE"):
            return 0

        if sys.version_info.major < 3:
            return string.atoi(self.__code.status["STATUS_RESULT_CODE"])
        return int(self.__code.status["STATUS_RESULT_CODE"])

    @property
    def result_desc(self):
        if not self.__code.status.has_key("STATUS_RESULT_DESC"):
            return ''
        if sys.version_info.major < 3:
            return string.atoi(self.__code.status["STATUS_RESULT_DESC"])
        return int(self.__code.status["STATUS_RESULT_DESC"])

    def put(self, vtype, name, value):
        oos = JceOutputStream()
        oos.write(vtype, 0, value)
        self.__buffer[name] = {vtype.__taf_class__: oos.getBuffer()}

    def get(self, vtype, name):
        if not self.__buffer.has_key(name):
            raise Exception("UniAttribute not found key:%s,type:%s" % (name, vtype.__taf_class__))
        t = self.__buffer[name]
        if not t.has_key(vtype.__taf_class__):
            raise Exception("UniAttribute not found type:" + vtype.__taf_class__)
        o = JceInputStream(t[vtype.__taf_class__])
        return o.read(vtype, 0, True)

    def encode(self):
        oos = JceOutputStream()
        oos.write(self.__mapv, 0, self.__buffer)
        self.__code.iVersion = 2
        self.__code.sBuffer = oos.getBuffer()
        sos = JceOutputStream()
        RequestPacket.writeTo(sos, self.__code)
        return struct.pack('!i', 4 + len(sos.getBuffer())) + sos.getBuffer()

    def decode(self, buf):
        ois = JceInputStream(buf[4:])
        self.__code = RequestPacket.readFrom(ois)
        sis = JceInputStream(self.__code.sBuffer)
        self.__buffer = sis.read(self.__mapv, 0, True)

    def clear(self):
        self.__code.__init__()

    def haskey(self, name):
        return self.__buffer.has_key(name)
