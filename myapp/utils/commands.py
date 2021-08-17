# -*- coding:utf-8 -*-
"""
执行本地命令
"""

import os
import subprocess
from . import commands
import traceback

class Command(object):
    """
    执行本地命令
    """

    @staticmethod
    def execute(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, use_async = False):
        """

        :param cmd:
        :param stdout:
        :param stderr:
        :param async:
        :return:
        """
        r_stdout = None
        r_stderr = None
        try:
            popen_obj = subprocess.Popen(cmd, stdin = None,
                                           stdout = stdout,
                                           stderr = stderr,
                                           shell  = True,
                                           close_fds=True);
            #非阻塞模式，直接返回
            if use_async:
                r_stdout, r_stderr = None, None
            else:
                r_stdout = popen_obj.stdout.read().strip()
                msg_stderr = popen_obj.stderr.read().strip()
                if not msg_stderr.strip():
                    r_stderr = msg_stderr
        except Exception as ex:
            r_stderr = traceback.format_exc()
        finally:
            try:
                if not use_async:
                    if popen_obj.stdout is not None:
                        popen_obj.stdout.close()

                    if popen_obj.stderr is not None:
                        popen_obj.stderr.close()

                    if popen_obj.stdin is not None:
                        popen_obj.stdin.close()
                    popen_obj.terminate()
            except:
                pass
        if r_stderr:
            raise Exception("exe command [%s] failed, %s" % (cmd, r_stderr))
        return r_stdout
    
    