
import datetime
import json
import os
import subprocess
import time

from job.pkgs.constants import ComponentOutput
from job.pkgs.context import JobComponentRunner
from job.pkgs.httpclients.model_repo_client import ModelRepoClient
from job.pkgs.tf.helperfuncs import try_get_model_name
from job.pkgs.utils import make_abs_or_pack_path, try_archive_by_config
from job.pkgs.utils import split_file_name


class PlainTrainer(JobComponentRunner):
    def job_func(self, jc_entry):
        print("TF_CONFIG={}".format(os.environ.get('TF_CONFIG')))

        job = jc_entry.job

        script_name = job.get('script_name', '').strip()
        if not script_name:
            raise RuntimeError("'script_name' not set")

        script = make_abs_or_pack_path(script_name.strip(), jc_entry.data_path, jc_entry.pack_path)
        if not os.path.isfile(script):
            raise RuntimeError("script '{}' not exist".format(script))
        workdir = os.path.dirname(script)
        script_params = job.get('params', [])
        script_params = list(map(str, script_params))
        
        st = time.time()
        print("{}: begin running training script '{}', params={}, cwd='{}'"
              .format(self.job_name, script, script_params, workdir))
        subprocess.check_call(["python3", script] + script_params, cwd=workdir)
        print("{}: training script '{}' finished, cost {}s".format(self.job_name, script, time.time() - st))

        if self.is_chief(json.loads(os.environ.get('TF_CONFIG') or '{}')):
            model_name = job.get('model_name', '').strip()
            save_path = job.get('save_path', '').strip()
            version = job.get('version', '').strip()
            if not version:
                version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")

            if os.path.isdir(save_path):
                if not model_name:
                    model_name = try_get_model_name(save_path) or split_file_name(script_name)[1]
                archive_config = {
                    "src": save_path,
                    "path_name": model_name
                }
                archived = try_archive_by_config(archive_config, jc_entry.data_path, jc_entry.pack_path)
                if archived:
                    save_path = archived[0][1]
                mrc = ModelRepoClient(jc_entry.creator)
                mrc.add_model(jc_entry.pipeline_id, jc_entry.run_id, 'tf', model_name, save_path, version,
                              None, thr_exp=False)
                print("{}: added model training record, pipeline_id='{}', run_id='{}', model_name='{}', save_path='{}'"
                      .format(self.job_name, jc_entry.pipeline_id, jc_entry.run_id, model_name, save_path))

                return '|'.join([save_path, model_name])
            else:
                print("WARNING: save path '{}' not exists".format(save_path))
        else:
            print("not chief")

    @classmethod
    def is_chief(cls, tf_config):
        if not tf_config:
            return True

        cluster = tf_config.get('cluster', {})
        role = tf_config.get('task', {}).get('type', '')
        index = tf_config.get('task', {}).get('index')
        if role:
            role = role.strip().lower()
        return role in ['chief', 'master'] or ('chief' not in cluster and 'master' not in cluster
                                               and role == 'worker' and index == 0)


if __name__ == "__main__":
    trainer = PlainTrainer("TF plain train component", ComponentOutput.MODEL_TRAIN_OUTPUT)
    trainer.run()
