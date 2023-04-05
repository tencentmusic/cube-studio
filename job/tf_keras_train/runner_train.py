# coding=utf-8
# @Time     : 2020/9/28 20:50
# @Auther   : lionpeng@tencent.com

import datetime
import json
import os

from job.pkgs.constants import ComponentOutput
from job.pkgs.context import JobComponentRunner
from job.pkgs.httpclients.model_repo_client import ModelRepoClient
from job.pkgs.tf.model_runner import TFModelRunner
from job.pkgs.utils import (make_abs_or_pack_path, split_file_name,
                            try_archive_by_config)


class TFRunnerTrainer(JobComponentRunner):
    def job_func(self, jc_entry):
        tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
        print("tf_config={}".format(tf_config))
        
        ps_config = json.loads(os.environ.get('PS_CONFIG') or '{}')
        print("ps_config={}".format(ps_config))
        if ps_config:
            tf_config['ps_config'] = ps_config

        job = jc_entry.job
        user_py_file = job.get('script_name')
        model_args = job.get('model_args', {})
        train_args = job.get('train_args', {})
        train_data_args = job.get('train_data_args', {})
        val_data_args = job.get('val_data_args', {})
        save_model_args = job.get('save_model_args', {})
        if not user_py_file or not user_py_file.strip():
            raise RuntimeError("'script_name' not set")
        user_py_file = make_abs_or_pack_path(user_py_file, jc_entry.export_path, jc_entry.pack_path)
        if not os.path.isfile(user_py_file):
            raise RuntimeError("user script '{}' not exist".format(user_py_file))

        workdir = os.getcwd()
        os.chdir(os.path.dirname(user_py_file))
        print("change work dir from '{}' to '{}'".format(workdir, os.getcwd()))

        runner = TFModelRunner(user_py_file, jc_entry.export_path, jc_entry.pack_path, tf_config,
                               model_args=model_args, train_args=train_args,
                               train_data_args=train_data_args,
                               val_data_args=val_data_args,
                               save_model_args=save_model_args)
        saved_models = runner.run_train()
        if runner.is_chief() and saved_models:
            version = train_args.get('version', '').strip()
            if not version:
                version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
            save_info_strings = []
            for model_save_path, model_name in saved_models:
                # if not model_name:
                #     model_name = split_file_name(user_py_file)[1]
                #     print("model name not set, will use script file base name '{}' as model name".format(model_name))

                archive_config = {
                    "src": model_save_path,
                    "path_name": model_name
                }
                archived = try_archive_by_config(archive_config, jc_entry.export_path, jc_entry.pack_path)
                if archived:
                    model_save_path = archived[0][1]
                mrc = ModelRepoClient(jc_entry.creator)
                mrc.add_model(jc_entry.pipeline_id, jc_entry.run_id, 'tf', model_name, model_save_path,
                              version, None, thr_exp=False)
                print("added model training record, pipeline_id='{}', run_id='{}', model_name='{}', mode_save_path='{}'"
                      .format(jc_entry.pipeline_id, jc_entry.run_id, model_name, model_save_path))
                save_info_strings.append('|'.join([model_save_path, model_name]))

            return '\n'.join(save_info_strings)
        else:
            print("not chief")


if __name__ == '__main__':
    trainer = TFRunnerTrainer("TF runner trainer component", ComponentOutput.MODEL_TRAIN_OUTPUT)
    trainer.run()
