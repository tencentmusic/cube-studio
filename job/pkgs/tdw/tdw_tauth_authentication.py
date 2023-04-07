import base64
import json
import random
import socket
import traceback
import uuid
from datetime import datetime as dt

import pyDes
import requests

DEV_BASE_URL = "http://10.240.137.24:8081"
PROD_BASE_URL = "http://auth.tdw.oa.com"


class TdwTauthAuthentication:

    def __init__(self, tdw_sec_file, target, proxyUser=None, env="prod", proxyServers=None, time_slack_ms=15000):
        super().__init__()
        with open(tdw_sec_file) as f:
            sec = json.load(f)
            self.username = sec['subject']
            self.cmk = sec['key']
        self.expire = True
        self.expireTimeStamp = int(dt.timestamp(dt.now()))*1000
        self.proxyUser = proxyUser
        self.ip = self.get_host_ip()
        self.sequence = random.randint(0,999)
        self.identifier = {
            "user":self.username,
            "host":self.ip,
            "target":target,
            "lifetime":7200000
        }

        self.ClientAuthenticator = {
            "principle":self.username,
            "host":self.ip
        }
        self.baseUrl = PROD_BASE_URL if env is None or env.lower() == 'prod' else DEV_BASE_URL
        self.proxyServers = proxyServers
        self.time_slack_ms = max(time_slack_ms, 0)

    def get_host_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception as e:
            print("failed to get host ip: {}\n{}".format(e, traceback.format_exc()))
            return "127.0.0.1"
        finally:
            s.close()
        return ip

    def getSessionTicket(self):
        requestUrl = self.baseUrl + "/api/auth/st2"
        try:
            self.identifier["timestamp"] = int(dt.timestamp(dt.now()))
            identifierBody = base64.b64encode(bytes(json.dumps(self.identifier), 'utf8'))
            response = requests.get(requestUrl, params={"ident": identifierBody}, proxies=self.proxyServers)
            self.sessionTicket = response.text
        except Exception as e:
            print("failed to get session ticket for user '{}', url='{}', proxies={}: {}\n{}"
                  .format(self.username, requestUrl, self.proxyServers, e, traceback.format_exc()))

    def decryptClientTicket(self):
        try:
            sessionTicket = json.loads(self.sessionTicket)
        except Exception as e:
            print("parse session ticket for user '{}' error, ticket='{}': {}\n{}"
                  .format(self.username, self.serviceTicket, e, traceback.format_exc()))
            raise e

        try:
            self.serviceTicket = sessionTicket["st"]
            clientTicket = sessionTicket["ct"]
            clientTicket = base64.b64decode(clientTicket)
            cmk = bytes.fromhex(base64.b64decode(self.cmk).decode('utf8'))
            DESede = pyDes.triple_des(cmk, pyDes.ECB, padmode=pyDes.PAD_PKCS5)
            clientTicket = json.loads(str(DESede.decrypt(clientTicket),'utf8'))
            self.expireTimeStamp = clientTicket["timestamp"] + clientTicket["lifetime"] - self.time_slack_ms
            self.sessionKey = base64.b64decode(clientTicket['sessionKey'])
        except Exception as e:
            print("decrypt clent key for user '{}' error: {}\n{}".format(self.username, e, traceback.format_exc()))
            raise e

    def constructAuthentication(self):
        try:
            self.ClientAuthenticator["timestamp"] = int(dt.timestamp(dt.now()))*1000
            self.ClientAuthenticator["nonce"] = uuid.uuid1().hex
            self.ClientAuthenticator["sequence"] = self.sequence
            self.sequence += 1
            if self.proxyUser:
                self.ClientAuthenticator["proxyUser"] = self.proxyUser
            ClientAuthenticator = bytes(json.dumps(self.ClientAuthenticator), 'utf8')

            DESede = pyDes.triple_des(self.sessionKey, pyDes.ECB, padmode=pyDes.PAD_PKCS5)
            ClientAuthenticator = DESede.encrypt(ClientAuthenticator)
            authentication = "tauth."+self.serviceTicket+"."+str(base64.b64encode(ClientAuthenticator), 'utf8')
            return {"secure-authentication": authentication}
        except Exception as e:
            print("construct authentication for user '{}' error: {}\n{}".format(self.username, e, traceback.format_exc()))
            raise e

    def isExpire(self):
        return self.expireTimeStamp <= int(dt.timestamp(dt.now()))*1000

    def getAuthentication(self):
        if self.isExpire():
            self.getSessionTicket()
            self.decryptClientTicket()
        return self.constructAuthentication()
