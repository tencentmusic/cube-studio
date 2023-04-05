# -*- encoding=utf-8 -*-
__author__ = "kevintian@tencnet.com (Kevin.Tian)"
__author__ = "kevintian@wuhan.tencent.com (Kevin.Tian)"
__version__ = "0.0.1"

from .__util import util
from .__jce import JceInputStream
from .__jce import JceOutputStream
from .__wup import TafUniPacket
import json


class tafcore:
    class JceInputStream(JceInputStream):
        pass

    class JceOutputStream(JceOutputStream):
        pass

    class TafUniPacket(TafUniPacket):
        pass

    class boolean(util.boolean):
        pass

    class int8(util.int8):
        pass

    class uint8(util.uint8):
        pass

    class int16(util.int16):
        pass

    class uint16(util.uint16):
        pass

    class int32(util.int32):
        pass

    class uint32(util.uint32):
        pass

    class int64(util.int64):
        pass

    class float(util.float):
        pass

    class double(util.double):
        pass

    class bytes(util.bytes):
        pass

    class string(util.string):
        pass

    class struct(util.struct):
        def toJSON(self):
            return json.dumps(self, default=lambda o: o.__dict__)

        def loads(self, dic):
            for k in self.__dict__:
                print(k)
                if issubclass(self.__dict__[k].__class__, list):
                    self.__dict__[k].loads(dic[k])
                else:
                    self.__dict__[k] = dic[k]

    @staticmethod
    def mapclass(ktype, vtype):
        return util.mapclass(ktype, vtype)

    @staticmethod
    def vctclass(vtype):
        return util.vectorclass(vtype)

    @staticmethod
    def printHex(buff):
        util.printHex(buff)
