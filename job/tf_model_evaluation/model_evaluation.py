# coding=utf-8
# @Time     : 2020/9/28 20:50
# @Auther   : lionpeng@tencent.com

import json
import os
from collections import defaultdict

from job.pkgs.constants import ComponentOutput, PipelineParam
from job.pkgs.context import JobComponentRunner
from job.pkgs.httpclients.model_repo_client import ModelRepoClient
from job.pkgs.tf.helperfuncs import try_get_model_name
from job.pkgs.tf.model_runner import TFModelRunner
from job.pkgs.utils import make_abs_or_data_path, make_abs_or_pack_path
from prettytable import PrettyTable


class TFModelEvaluator(JobComponentRunner):
    def job_func(self, jc_entry):
        tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
        print("tf_config={}".format(tf_config))

        job = jc_entry.job.get('job_detail', {}) or jc_entry.job
        user_py_file = job.get('script_name')
        train_args = job.get('train_args', {})
        model_args = job.get('model_args', {})
        evaluation_args = job.get('eval_args') or job.get('test_args') or {}
        test_data_args = job.get('eval_data_args') or job.get('test_data_args') or {}
        load_model_args = job.get('load_model_args', {})

        if not user_py_file or not user_py_file.strip():
            raise RuntimeError("'script_name' not set")
        user_py_file = make_abs_or_pack_path(user_py_file, jc_entry.export_path, jc_entry.pack_path)
        if not os.path.isfile(user_py_file):
            raise RuntimeError("user script '{}' not exist".format(user_py_file))

        if not evaluation_args:
            raise RuntimeError("'evaluation_args' not set")

        upstream_models = []
        if jc_entry.upstream_output_file:
            if not os.path.isfile(jc_entry.upstream_output_file):
                print("{}: upstream output file '{}' not exist, ignore it"
                      .format(self.job_name, jc_entry.upstream_output_file))
            else:
                with open(jc_entry.upstream_output_file) as f:
                    i = 0
                    for line in f.readlines():
                        if not line or not line.strip():
                            continue
                        line = line.strip()
                        fields = line.split('|')
                        up_model_path = fields[0].strip()
                        if not up_model_path or not os.path.exists(up_model_path):
                            print("{}: model path '{}' from upstream output file not set or exist"
                                  .format(self.job_name, up_model_path))
                            continue
                        if len(fields) == 2:
                            up_model_name = fields[1].strip()
                        else:
                            up_model_name = ''
                        if not up_model_name:
                            up_model_name = try_get_model_name(up_model_path) or 'upstream_model_%s' % i
                            print("{}: model name not set in upstream output file '{}', set model name as '{}'"
                                  .format(self.job_name, jc_entry.upstream_output_file, up_model_name))
                        upstream_models.append({'path': up_model_path, 'name': up_model_name})
                        print("{}: add upstream model '{}' in '{}' from upstream output file '{}'"
                              .format(self.job_name, up_model_name, up_model_path, jc_entry.upstream_output_file))
                        i += 1

                upstream_models.extend(evaluation_args.get('models', []))
                evaluation_args['models'] = upstream_models

        candi_models = evaluation_args.get('models', [])
        if isinstance(candi_models, (str, dict)):
            candi_models = [candi_models]
        elif not isinstance(candi_models, list):
            raise RuntimeError("'models' should be a list or dict or str, got '{}': {}"
                               .format(type(candi_models), candi_models))

        def __get_online_model_spec():
            spec = ModelRepoClient(jc_entry.creator).get_online_model_info(jc_entry.pipeline_id, thr_exp=False)
            if not spec:
                print("{}: found no online model of pipeline_id '{}', ignore it"
                      .format(self.job_name, jc_entry.pipeline_id))
            else:
                print("{}: found online model of pipeline_id '{}': {}"
                      .format(self.job_name, jc_entry.pipeline_id, spec))
            return spec

        expanded_candi_models = []
        for i, cm in enumerate(candi_models):
            expanded_cm = {}
            if isinstance(cm, str):
                cm = cm.strip()
                if not cm:
                    print("{}: {}th model in models is empty, ignore it".format(self.job_name, i))
                    continue
                if cm == PipelineParam.ONLINE_MODEL:
                    online_model_spec = __get_online_model_spec()
                    if not online_model_spec:
                        continue
                    expanded_cm['name'] = online_model_spec['model_name']
                    expanded_cm['path'] = online_model_spec['model_path']
                else:
                    expanded_cm['path'] = cm
            elif isinstance(cm, dict):
                model_path = cm.get('path', '').strip()
                if not model_path:
                    print("{}: 'path' of {}th model in models not set, ignore it: {}".format(self.job_name, i, cm))
                    continue
                if model_path == PipelineParam.ONLINE_MODEL:
                    online_model_spec = __get_online_model_spec()
                    if not online_model_spec:
                        continue
                    expanded_cm['path'] = online_model_spec['model_path']
                    if 'name' not in cm:
                        expanded_cm['name'] = online_model_spec['model_name']
                else:
                    expanded_cm['path'] = model_path
                    expanded_cm['name'] = cm.get('name', '')
            else:
                raise RuntimeError("{}th model in models should be a dict or str, got {}: {}".format(i, type(cm), cm))

            if not os.path.exists(expanded_cm['path']):
                print("{}: {}th model path '{}' not exists, ignore it".format(self.job_name, i, expanded_cm['path']))
                continue
            model_name = expanded_cm.get('name', '').strip()
            if not model_name:
                expanded_cm['name'] = try_get_model_name(expanded_cm['path']) or 'eval_model_{}'.format(i)
            expanded_candi_models.append(cm)

        evaluation_args['models'] = expanded_candi_models
        print("{}: expaneded candidate models: {}".format(self.job_name, expanded_candi_models))

        runner = TFModelRunner(user_py_file, jc_entry.export_path, jc_entry.pack_path, train_args=train_args,
                               model_args=model_args, evaluate_args=evaluation_args, 
                               test_data_args=test_data_args, load_model_args=load_model_args, 
                               tf_config=tf_config)
        eval_results = runner.run_evaluate()
        if eval_results and runner.is_chief():
            table = PrettyTable()
            table.add_column('ModelName', [name for name, _, _ in eval_results])
            index_keys = eval_results[0][2].keys()
            for k in index_keys:
                table.add_column(k, [eval_ret.get(k) for _, _, eval_ret in eval_results])
            table.add_column('ModelPath', [path for _, path, _ in eval_results])

            print("{}: model evaluation results:\n".format(self.job_name))
            print(table)

            output = {'eval_results': eval_results}
            extra_output_file = evaluation_args.get('output_file', '').strip()
            if extra_output_file:
                extra_output_file = make_abs_or_data_path(extra_output_file, jc_entry.data_path, jc_entry.pack_path)
                with open(extra_output_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(output))
                    print("{}: wrote outputs into extra output file '{}': {}"
                          .format(self.job_name, extra_output_file, output))

            if upstream_models:
                upstream_model_name = upstream_models[0]['name']
                upstream_model_path = upstream_models[0]['path']
                upstream_model_metric = {}
                for er in eval_results:
                    if er[0] == upstream_model_name and er[1] == upstream_model_path:
                        upstream_model_metric = er[2]
                        break
                updates = {"metrics": json.dumps(upstream_model_metric)}
                ModelRepoClient(jc_entry.creator).update_model(jc_entry.pipeline_id, jc_entry.run_id,
                                                               updates, thr_exp=False)
                print("{}: updated model metrics, pipeline_id='{}', run_id='{}', model_name='{}', model_path='{}',"
                      " updates: {}".format(self.job_name, jc_entry.pipeline_id, jc_entry.run_id, upstream_model_name,
                                            upstream_model_path, updates))
            return output
        else:
            print("{}: got no model evaluation result, is_chief={}".format(self.job_name, runner.is_chief()))


if __name__ == '__main__':
    evaluator = TFModelEvaluator("TF model evaluator component", ComponentOutput.MODEL_EVALUATION_OUTPUT)
    evaluator.run()
