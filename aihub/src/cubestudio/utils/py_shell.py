
import os,sys,subprocess

def exec(command):
    # os.system('bash %s' % command)

    cmd = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,stdout=subprocess.PIPE, universal_newlines=True, shell=True, bufsize=1)