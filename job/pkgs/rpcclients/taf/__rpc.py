# -*- encoding=utf-8 -*-
import socket
import struct

from .__jce import JceInputStream
from .__jce import JceOutputStream
from .__packet import RequestPacket
from .__packet import ResponsePacket
from .__util import util


class ServantProxy(object):
    JCEVERSION = 1
    JCENORMAL = 0x0
    JCEMESSAGETYPENULL = 0x0
    JCESERVERSUCCESS = 0
    mapcls_context = util.mapclass(util.string, util.string)

    def __init__(self):
        self.ip = ''
        self.port = 0
        self.sServantName = ''
        self.iTimeout = 3000

    def locator(self, connInfo):
        self.sServantName, argvs = connInfo.split('@')
        args = argvs.lower().split()
        if len(args) == 0:
            raise Exception('string parsing error around "@"')
        if args[0] != 'tcp':
            raise Exception('unsupport transmission protocal : %s' % args[0])
        i = 1
        while i < len(args):
            if args[i] == '-h':
                i += 1
                self.ip = args[i]
            elif args[i] == '-p':
                i += 1
                self.port = int(args[i])
            else:
                raise Exception('unkown parameter : %s' % args[i])
            i += 1
        if self.ip == '' or self.port == 0:
            raise Exception('can not find ip or port info')

    def taf_invoke(self, cPacketType, sFuncName, sBuffer, context, status):
        req = RequestPacket()
        req.iVersion = ServantProxy.JCEVERSION
        req.cPacketType = cPacketType
        req.iMessageType = ServantProxy.JCEMESSAGETYPENULL
        req.iRequestId = 0
        req.sServantName = self.sServantName
        req.sFuncName = sFuncName
        req.sBuffer = sBuffer
        req.iTimeout = self.iTimeout

        oos = JceOutputStream()
        RequestPacket.writeTo(oos, req)

        reqpkt = oos.getBuffer()
        plen = len(reqpkt) + 4
        reqpkt = struct.pack('!i', plen) + reqpkt

        ret = self.__trans(reqpkt, plen)
        if len(ret) == 0:
            raise Exception('server do not response')

        ios = JceInputStream(ret)
        rsp = ResponsePacket.readFrom(ios)
        if rsp.iRet != 0:
            raise Exception("Taf Error:%d" % rsp.iRet)
        return rsp

    # send
    def __trans(self, buf, blen):
        try:
            s = socket.socket()
            s.connect((self.ip, self.port))
            s.send(buf)
            ret = s.recv(4)
            blen, = struct.unpack_from('!i', ret)
            blen -= 4
            # ret = s.recv(blen)
            # Modified by jasonling 
            bufSize = 1024
            total = 0
            ret = b''
            while total < blen:
                tmpStr = s.recv(bufSize)
                total += len(tmpStr)
                # print total
                ret += tmpStr
            # print str(len(ret)) + '_' + str(blen)
            if len(ret) != blen:
                raise Exception('receive response packet error')
            s.close()
        except socket.error as e:
            raise Exception(e)

        return ret
