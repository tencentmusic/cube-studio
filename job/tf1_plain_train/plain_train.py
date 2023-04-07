# -*- coding: utf-8 -*-
# @Time     : 2020/08/28 20:50
# @Auther   : lionpeng@tencent.com

import argparse
import json
import os
import subprocess
import time
import datetime

from job.pkgs.constants import ComponentOutput
from job.pkgs.context import KFJobContext
from job.pkgs.httpclients.model_repo_client import ModelRepoClient
from job.pkgs.utils import make_abs_or_data_path, make_abs_or_pack_path, expand_param, recur_expand_param, \
    try_archive_by_config
from job.pkgs.utils import split_file_name
from job.pkgs.datapath_cleaner import DataPathCleaner
from job.pkgs.tf.helperfuncs import try_get_model_name


def should_write_output_file():
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    if not tf_config:
        return True

    cluster = tf_config.get('cluster', {})
    role = tf_config.get('task', {}).get('type', '')
    index = tf_config.get('task', {}).get('index')
    if role:
        role = role.strip().lower()
    return role in ['chief', 'master'] or ('chief' not in cluster and 'master' not in cluster and role == 'worker' and index == 0)


def model_train(pack_path, export_path, script_name, upstream_output_file, pipeline_id, run_id,
                params, output_file, save_path, creator, clean_files, version):
    """
    执行模型脚本，output_file的路径名会作为本脚本的第一个参数传入，模型输出相关文件需放在这个路径下
    Args:
        pack_path (str): 用户包挂载到容器中的路径
        export_path (str): 用户数据目录
        script_name (str): 脚本文件名
        upstream_output_file (str): 上游输出文件名，会从上游输出文件中解析出本脚本需要的参数信息
        pipeline_id (str): pipeline id
        run_id (str): 运行id
        params (list): 脚本的参数
        output_file (str): 数据转换完成后数据文件名(包含路径)
        save_path (str): 模型导出目录
        creator (str): pipeline创建者
        clean_files (str or list, optional): 完成后要清理的文件
        version (str): 模型版本

    Returns:

    """
    if not pack_path or not pack_path.strip():
        pack_path = os.path.abspath('.')

    if not script_name or not script_name.strip():
        raise RuntimeError("'script_name' not set")
    script = make_abs_or_pack_path(script_name.strip(), export_path, pack_path)
    if not os.path.isfile(script):
        raise RuntimeError("script '{}' not exist".format(script))
    workdir = os.path.dirname(script)

    params = params or []
    if upstream_output_file:
        if not os.path.isfile(upstream_output_file):
            print("upstream_output_file '{}' not exist, ignore it".format(upstream_output_file))
        else:
            upstream_params = []
            if upstream_output_file and upstream_output_file.strip():
                with open(upstream_output_file.strip()) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        for field in line.split('|'):
                            if not field.strip():
                                continue
                            upstream_params.append(field.strip())
                params.extend(upstream_params)
                print("read params from upstream output file '{}': {}".format(upstream_output_file, upstream_params))
            else:
                print("no params from upstream")

    params = list(map(lambda p: str(expand_param(p, export_path, pack_path)), params))
    st = time.time()
    print("begin running training script '{}', params={}, cwd='{}'".format(script, params, workdir))
    subprocess.check_call(["python3", script] + params, cwd=workdir)
    print("training script '{}' finished, cost {}s".format(script, time.time() - st))

    if should_write_output_file():
        if os.path.isdir(save_path):
            mrc = ModelRepoClient(creator)
            model_name = try_get_model_name(save_path) or split_file_name(script_name)[1]
            archive_config = {
                "src": save_path,
                "path_name": model_name
            }
            archived = try_archive_by_config(archive_config, export_path, pack_path)
            if archived:
                save_path = archived[0][1]
            mrc.add_model(pipeline_id, run_id, 'tf', model_name, save_path, version, None, thr_exp=False)
            # print("added model training record, pipeline_id='{}', run_id='{}', model_name='{}', save_path='{}'"
            #       .format(pipeline_id, run_id, model_name, save_path))

            with open(output_file, 'w') as f:
                f.write('|'.join([save_path, model_name]) + "\n")
                print("wrote model '{}' save path '{}' into '{}'".format(model_name, save_path, output_file))
        else:
            print("save path '{}' not exists, will not write into output file '{}'"
                  .format(save_path, output_file))
        DataPathCleaner().clean(clean_files)
    else:
        print("not chief, will not write model save path '{}' into '{}'".format(save_path, output_file))


def main(job, pack_path, upstream_output_file, export_path, pipeline_id, run_id, creator):
    ctx = KFJobContext.get_context()
    print("ctx: {}".format(ctx))

    pack_path = pack_path or ctx.pack_path
    export_path = export_path or ctx.export_path
    pipeline_id = pipeline_id or ctx.pipeline_id
    run_id = run_id or ctx.run_id
    creator = creator or ctx.creator
    job = recur_expand_param(job, export_path, pack_path)
    print("expanded job spec: {}".format(job))

    print("TF_CONFIG={}".format(os.environ.get('TF_CONFIG')))

    if job:
        output_file = os.path.join(export_path, ComponentOutput.MODEL_TRAIN_OUTPUT)
        if upstream_output_file and upstream_output_file.strip():
            upstream_output_file = os.path.join(export_path, os.path.basename(upstream_output_file.strip()))
        else:
            upstream_output_file = None
            print("upstream_output_file not set")

        save_path = job.get('save_path', '').strip()
        version = job.get('version', '').strip()
        if not version:
            version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
        script_params = job.get('params', [])
        if save_path:
            if save_path not in script_params:
                print("WARNING: 'save_path'='{}' not present in script params: {}".format(save_path, script_params))
            save_path = make_abs_or_data_path(save_path, export_path, pack_path)
            if not os.path.isdir(save_path):
                os.makedirs(save_path, exist_ok=True)
                print("{}: created model save path '{}'".format(__file__, save_path))
        else:
            print("WARNING: save_path not set")

        model_train(pack_path, export_path, job.get('script_name'), upstream_output_file, pipeline_id, run_id,
                    script_params, output_file, save_path, creator, job.get('clean_files'), version)
    else:
        print("empty model train job")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("model plain train component")
    arg_parser.add_argument('--job', type=str, required=True, help="模型训练任务描述json")
    arg_parser.add_argument('--pack-path', type=str, help="用户包（包含所有用户文件的目录）的挂载到容器中的路径")
    arg_parser.add_argument('--upstream-output-file', type=str, help="上游输出文件（包含路径）")
    arg_parser.add_argument('--export-path', type=str, help="数据导出目录")
    arg_parser.add_argument('--pipeline-id', type=str, help="pipeline id")
    arg_parser.add_argument('--run-id', type=str, help="运行id，标识每次运行")
    arg_parser.add_argument('--creator', type=str, help="pipeline的创建者")

    args = arg_parser.parse_args()
    print("{} args: {}".format(__file__, args))

    job_spec = json.loads(args.job)
    main(job_spec, args.pack_path, args.upstream_output_file, args.export_path, args.pipeline_id,
         args.run_id, args.creator)
