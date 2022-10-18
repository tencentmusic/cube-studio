
# # https://dormousehole.readthedocs.io/en/latest/cli.html
# @cli.command('init')
# # @pysnooper.snoop()
# def debug():
#     pass

import argparse
import gettext
import io
import json
import logging
import os
import re
import sys
import time
import click

@click.command()
@click.argument('ops', required=True)
@click.option('-p', '--port', default=8080, help='暴露端口')
def main(ops,port):
    if ops=='debug':
        command ='docker run --name aihub --privileged --rm -it -v $PWD:/app -p 8080:8080 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/aihub:base bash'
        print('command')

    print('hello from debug')


if __name__ == "__main__":
    main()