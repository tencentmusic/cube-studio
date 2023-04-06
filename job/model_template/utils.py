# coding=utf-8
# @Time     : 2021/1/17 11:49
# @Auther   : lionpeng@tencent.com
import copy
import json
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor

from job.pkgs.constants import ComponentOutput
from job.pkgs.context import JobComponentEntry, JobComponentRunner
from job.pkgs.utils import recur_expand_param
from job.tf_distributed_evaluation.tfjob_launcher import \
    TFJobLauncher as EvalTFJobLauncher
from job.tf_distributed_train.tfjob_launcher import \
    TFJobLauncher as TrainTFJobLauncher


def replace_default_config(default_config, update_config, base_path='', ignored_keys=None):
    if not default_config:
        return update_config
    if not update_config:
        return default_config
    if not isinstance(default_config, dict) or not isinstance(update_config, dict):
        return update_config

    output_config = {}
    for default_key, default_value in default_config.items():
        output_config[default_key] = default_value

    for update_key, update_value in update_config.items():
        qualified_key = base_path + '/' + update_key
        if ignored_keys and qualified_key in ignored_keys:
            print("key '{}' in ignored_keys {}, will not overide default value: {}"
                  .format(qualified_key, ignored_keys, default_config.get(update_key)))
            continue
        if not isinstance(update_value, dict):
            default_value = default_config.get(update_key)
            output_config[update_key] = update_value
            print("override '{}' from '{}' to '{}'".format(qualified_key, default_value, update_value))
        else:
            output_config[update_key] = replace_default_config(default_config.get(update_key, {}), update_value,
                                                               qualified_key, ignored_keys)
    return output_config


def local_train_func(job_name, job, pack_path, export_path, output_file):
    # import tensorflow as tf
    # tf.config.run_functions_eagerly(True)
    print("local train process id={}".format(os.getpid()))
    from job.tf_keras_train.runner_train import TFRunnerTrainer
    trainer = TFRunnerTrainer(job_name + " trainer", args=[
        "--job", json.dumps(job),
        "--pack-path", pack_path,
        "--export-path", export_path,
        "--output-file", output_file
    ])
    trainer.run()


def local_eval_func(job_name, job, pack_path, export_path, train_output_file, output_file):
    # import tensorflow as tf
    # tf.config.run_functions_eagerly(True)
    from job.tf_model_evaluation.model_evaluation import TFModelEvaluator
    evaluator = TFModelEvaluator(job_name + " model evaluator", args=[
        "--job", json.dumps(job),
        "--pack-path", pack_path,
        "--export-path", export_path,
        "--upstream-output-file", train_output_file,
        "--output-file", output_file
    ])
    evaluator.run()


def predict_func(job_name, job, pack_path, export_path, train_output_file):
    # import tensorflow as tf
    # tf.config.run_functions_eagerly(True)
    from job.tf_model_offline_predict.main import TFModelOfflinePredictor
    predictor = TFModelOfflinePredictor(job_name + " model offline predictor", args=[
        "--job", json.dumps(job),
        "--pack-path", pack_path,
        "--export-path", export_path,
        "--upstream-output-file", train_output_file
    ])
    predictor.run()


DEF_MODEL_TEMP_FIX_CONFIG_KEYS = {'/trainer', '/job_detail/script_name'}


class ModelTemplateDriver(JobComponentRunner):
    def __init__(self, job_name, output_file=None, args=None, local=False, driver_path=None,
                 config_fix_keys=None):
        super(ModelTemplateDriver, self).__init__(job_name, output_file, args)
        self._local = local
        if not driver_path:
            import inspect
            caller_file = inspect.currentframe().f_back.f_code.co_filename
            driver_path = os.path.dirname(os.path.abspath(caller_file))
        self._driver_path = driver_path
        print("{}: driver_path='{}'".format(self.job_name, self._driver_path))
        self._config_fix_keys = DEF_MODEL_TEMP_FIX_CONFIG_KEYS
        if config_fix_keys:
            self._config_fix_keys |= set(config_fix_keys)
        print("{}: config_fix_keys={}".format(self.job_name, self._config_fix_keys))

    @property
    def job_name(self) -> str:
        if self._local:
            return super().job_name + '(local)'
        return super().job_name

    @property
    def is_local(self) -> bool:
        return self._local is True

    def post_proc(self, jc_entry: JobComponentEntry, train_output_file: str):
        pass

    def _load_default_config(self, pack_path, data_path):
        cfg_file = os.path.join(self._driver_path, "config_template.json")
        template_py_file = os.path.join(self._driver_path, "template.py")

        with open(cfg_file, 'r') as f:
            def_config = json.load(f)
            def_config = recur_expand_param(def_config, data_path, pack_path)
            def_config['job_detail']['script_name'] = template_py_file
            print("loaded default job config from file '{}', set script_name='{}'"
                  .format(cfg_file, template_py_file))
            return def_config

    def on_parsed_job(self, job: dict, pack_path: str, data_path: str) -> dict:
        def_job = self._load_default_config(pack_path, data_path)
        final_job = replace_default_config(def_job, job, ignored_keys=self._config_fix_keys)
        job_detail = final_job['job_detail']

        model_input_config_file = job_detail.get('model_input_config_file')
        if not model_input_config_file or not os.path.isfile(model_input_config_file):
            raise RuntimeError("model_input_config_file '{}' not found".format(model_input_config_file))

        if 'model_args' in job_detail:
            job_detail['model_args']['model_input_config_file'] = model_input_config_file
            job_detail['model_args']['pack_path'] = pack_path
            job_detail['model_args']['data_path'] = data_path

        if 'train_data_args' in job_detail:
            train_data_input_config = job_detail['train_data_args'].get('model_input_config_file', '').strip()
            if not train_data_input_config:
                job_detail['train_data_args']['model_input_config_file'] = model_input_config_file

        if 'test_data_args' in job_detail:
            test_data_input_config = job_detail['test_data_args'].get('model_input_config_file', '').strip()
            if not test_data_input_config:
                job_detail['test_data_args']['model_input_config_file'] = model_input_config_file

        if 'val_data_args' in job_detail:
            val_data_input_config = job_detail['val_data_args'].get('model_input_config_file', '').strip()
            if not val_data_input_config:
                job_detail['val_data_args']['model_input_config_file'] = model_input_config_file

        if 'predict_data_args' in job_detail:
            predict_data_input_config = job_detail['predict_data_args'].get('model_input_config_file', '').strip()
            if not predict_data_input_config:
                job_detail['predict_data_args']['model_input_config_file'] = model_input_config_file

        job_detail['load_model_args'] = {'model_input_config_file': model_input_config_file}

        print("{}: final job: {}".format(self.job_name, final_job))
        return final_job

    def _train_proc(self, jc_entry: JobComponentEntry) -> str:
        job_detail = jc_entry.job['job_detail']
        load_model_path = job_detail.get('load_model_from', '').strip()
        comp_name = re.sub(r'\s+', '_', self.job_name)
        if (not load_model_path or not os.path.exists(load_model_path)) and job_detail.get('train_args') \
                and job_detail.get('train_data_args'):
            if self.is_local:
                model_output_file = ComponentOutput.MODEL_TRAIN_OUTPUT + "-" + comp_name + "_local"
                ppool = ProcessPoolExecutor(1)
                print("{}: start local training process".format(self.job_name))
                ppool.submit(local_train_func, self.job_name, job_detail, jc_entry.pack_path,
                             jc_entry.export_path, model_output_file).result()
                print("{}: local training process finished".format(self.job_name))
                ppool.shutdown(wait=True)
            else:
                model_output_file = ComponentOutput.MODEL_TRAIN_OUTPUT + "-" + comp_name + "_tfjob"
                tfjob_launcher = TrainTFJobLauncher(self.job_name + " Train TFjob launcher", args=[
                    "--job", json.dumps(jc_entry.job),
                    "--pack-path", jc_entry.pack_path,
                    "--export-path", jc_entry.export_path,
                    "--output-file", model_output_file
                ])
                tfjob_launcher.run()
        else:
            print("{}: skip training step".format(self.job_name))
            model_output_file = ComponentOutput.MODEL_TRAIN_OUTPUT + "-" + comp_name + "_load"
            with open(os.path.join(jc_entry.data_path, model_output_file), 'w') as of:
                of.write(os.path.abspath(load_model_path))

        return model_output_file

    def _eval_proc(self, jc_entry: JobComponentEntry, train_output_file: str):
        job_detail = jc_entry.job['job_detail']
        eval_args = job_detail.get('eval_args') or job_detail.get('test_args')
        if eval_args and (job_detail.get('test_data_args') or job_detail.get('eval_data_args')):
            try:
                if self.is_local:
                    eval_output_file = eval_args.get('output_file', '').strip()
                    if not eval_output_file:
                        comp_name = re.sub(r'\s+', '_', self.job_name)
                        eval_output_file = ComponentOutput.MODEL_EVALUATION_OUTPUT + "-" + comp_name + "_local"
                    ppool = ProcessPoolExecutor(1)
                    print("{}: start evaluation process".format(self.job_name))
                    ppool.submit(local_eval_func, self.job_name, jc_entry.job, jc_entry.pack_path, jc_entry.export_path,
                                 train_output_file, eval_output_file).result()
                    print("{}: evaluation process finished".format(self.job_name))
                    ppool.shutdown(wait=True)
                else:
                    eval_output_file = eval_args.get('output_file', '').strip()
                    if not eval_output_file:
                        comp_name = re.sub(r'\s+', '_', self.job_name)
                        eval_output_file = ComponentOutput.MODEL_EVALUATION_OUTPUT + "-" + comp_name + "_tfjob"

                    eval_job = copy.deepcopy(jc_entry.job)
                    if eval_args.get('num_test_samples', 0) <= 0:
                        print("{}: 'num_test_samples' is not set, auto fallback to 1-worker evaluation"
                              .format(self.job_name))
                        eval_job['num_workers'] = 1
                    tfjob_launcher = EvalTFJobLauncher(self.job_name + " Evaluation TFjob launcher", args=[
                        "--job", json.dumps(eval_job),
                        "--pack-path", jc_entry.pack_path,
                        "--export-path", jc_entry.export_path,
                        "--output-file", eval_output_file,
                        "--upstream-output-file", train_output_file
                    ])
                    tfjob_launcher.run()
            except Exception as e:
                print("{}: WARING: evaluation model failed: {}\n{}".format(self.job_name, e, traceback.format_exc()))
        else:
            print("{}: skip evaluation step".format(self.job_name))

    def _predict_proc(self, jc_entry: JobComponentEntry, train_output_file: str):
        job_detail = jc_entry.job['job_detail']
        if job_detail.get('pred_data_args', job_detail.get('predict_data_args')) and \
                job_detail.get('pred_args', job_detail.get('predict_args')):
            try:
                ppool = ProcessPoolExecutor(1)
                print("{}: start prediction process".format(self.job_name))
                ppool.submit(predict_func, self.job_name, jc_entry.job, jc_entry.pack_path, jc_entry.export_path,
                             train_output_file).result()
                print("{}: prediction process finished".format(self.job_name))
                ppool.shutdown(wait=True)
            except Exception as e:
                print("{}: WARING: prediction failed: {}\n{}".format(self.job_name, e, traceback.format_exc()))
        else:
            print("{}: skip prediction step".format(self.job_name))

    def job_func(self, jc_entry: JobComponentEntry):
        train_output_file = self._train_proc(jc_entry)
        self._eval_proc(jc_entry, train_output_file)
        self._predict_proc(jc_entry, train_output_file)
        outputs = self.post_proc(jc_entry, train_output_file)
        return outputs
