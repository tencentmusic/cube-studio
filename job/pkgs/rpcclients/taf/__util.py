# -*- encoding=utf-8 -*-
__author__ = "kevintian@tencnet.com (Kevin.Tian)"
__author__ = "kevintian@wuhan.tencent.com (Kevin.Tian)"

import sys


class util:
    @staticmethod
    def printHex(buff):
        count = 0
        for c in buff:
            sys.stdout.write("0X%02X " % ord(c))
            count += 1
            if count % 16 == 0:
                sys.stdout.write("\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    @staticmethod
    def mapclass(ktype, vtype):
        class mapklass(dict):
            def size(self): return len(self)

        setattr(mapklass, '__taf_index__', 8)
        setattr(mapklass, '__taf_class__', "map<" + ktype.__taf_class__ + "," + vtype.__taf_class__ + ">")
        setattr(mapklass, 'ktype', ktype)
        setattr(mapklass, 'vtype', vtype)
        return mapklass

    @staticmethod
    def vectorclass(vtype):
        class klass(list):
            def size(self):
                return len(self)

            def loads(self, lst):
                for i in lst:
                    if vtype == util.struct:
                        tmp = vtype()
                        tmp.loads(i)
                        self.append(tmp)
                    else:
                        self.append(i)

        setattr(klass, '__taf_index__', 9)
        setattr(klass, '__taf_class__', "list<" + vtype.__taf_class__ + ">")
        setattr(klass, 'vtype', vtype)
        return klass

    class boolean:
        __taf_index__ = 999
        __taf_class__ = "bool"

    class int8:
        __taf_index__ = 0
        __taf_class__ = "char"

    class uint8:
        __taf_index__ = 1
        __taf_class__ = "short"

    class int16:
        __taf_index__ = 1
        __taf_class__ = "short"

    class uint16:
        __taf_index__ = 2
        __taf_class__ = "int32"

    class int32:
        __taf_index__ = 2
        __taf_class__ = "int32"

    class uint32:
        __taf_index__ = 3
        __taf_class__ = "int64"

    class int64:
        __taf_index__ = 3
        __taf_class__ = "int64"

    class float:
        __taf_index__ = 4
        __taf_class__ = "float"

    class double:
        __taf_index__ = 5
        __taf_class__ = "double"

    class bytes:
        __taf_index__ = 13
        __taf_class__ = "list<char>"

    class string:
        __taf_index__ = 67
        __taf_class__ = "string"

    class struct:
        __taf_index__ = 1011
