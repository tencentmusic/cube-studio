#!/bin/bash
import argparse,os,sys,io,time
import datetime
import shutil

config = open('to-csv.json').read()

from subprocess import Popen, PIPE, STDOUT

def exe_command(command):
    """
    执行 shell 命令并实时打印输出
    :param command: shell 命令
    :return: process, exitcode
    """
    print(command)
    process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            print(line.decode().strip(),flush=True)
    exitcode = process.wait()
    return exitcode

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("数据导入")
    arg_parser.add_argument('--db_type', type=str, help="数据库类型", default='')
    arg_parser.add_argument('--host', type=str, help="地址", default='')
    arg_parser.add_argument('--database', type=str, help="数据库", default='')
    arg_parser.add_argument('--table', type=str, help="表名", default='')
    arg_parser.add_argument('--columns', type=str, help="字段名", default='')
    arg_parser.add_argument('--query_sql', type=str, help="查询语句", default='')
    arg_parser.add_argument('--username', type=str, help="用户名", default='')
    arg_parser.add_argument('--password',type=str,help="密码",default='')
    arg_parser.add_argument('--save_path', type=str, help="保存地址", default='')

    args = arg_parser.parse_args()
    columns = [x.strip() for x in args.columns.split(',') if x.strip()]
    outdir = os.path.dirname(args.save_path)
    outname = os.path.basename(args.save_path)

    config = config.replace('READER',args.db_type)
    config = config.replace('USERNAME',args.username)
    config = config.replace('PASSWORD',args.password)
    config = config.replace('HOST',args.host)
    config = config.replace('DATABASE',args.database)
    config = config.replace('TABLE', args.table)
    config = config.replace('COLUMNS', str(columns))
    config = config.replace('OUTDIR',outdir)
    config = config.replace('OUTNAME', outname)
    config_file = args.db_type+'_csv.json'
    file = open(config_file,mode='w')
    file.write(config)
    file.close()

    command = f'python datax.py {config_file}'
    exitcode = exe_command(command)
    if exitcode:
        exit(exitcode)
    # 将导出文件的随机数去掉
    files = os.listdir(outdir)
    for file in files:
        if outname in file:
            file_stat = os.stat(os.path.join(outdir,file))
            creation_time = file_stat.st_ctime
            dt_object = datetime.datetime.fromtimestamp(creation_time)
            if (datetime.datetime.now()-dt_object).total_seconds()<10:
                if os.path.exists(args.save_path):
                    os.remove(args.save_path)
                shutil.move(os.path.join(outdir,file),args.save_path)










