
import os
from job.pkgs.context import JobComponentRunner
from job.pkgs.tf.model_runner import TFModelRunner
from job.pkgs.utils import make_abs_or_pack_path
from job.pkgs.tf.helperfuncs import try_get_model_name


class TFModelOfflinePredictor(JobComponentRunner):
    def job_func(self, jc_entry):
        job = jc_entry.job.get('job_detail', {})
        user_py_file = job.get('script_name', '').strip()
        predict_args = job.get('predict_args') or job.get('pred_args') or {}
        predict_data_args = job.get('predict_data_args') or job.get('pred_data_args') or {}

        if not user_py_file:
            raise RuntimeError("'script_name' not set")
        user_py_file = make_abs_or_pack_path(user_py_file, jc_entry.export_path, jc_entry.pack_path)
        if not os.path.isfile(user_py_file):
            raise RuntimeError("user script '{}' not exist".format(user_py_file))

        model_path = predict_args.get('model_path', '').strip()
        if model_path and os.path.exists(model_path):
            print("{}: user set 'model_path' '{}'".format(self.job_name, model_path))
        elif jc_entry.upstream_output_file:
            print("{}: 'model_path'='{}' not set or not exists, will resort to upstrem file '{}'"
                  .format(self.job_name, model_path, jc_entry.upstream_output_file))

            if not os.path.isfile(jc_entry.upstream_output_file):
                raise RuntimeError("upstream output file '{}' not exist".format(jc_entry.upstream_output_file))
            else:
                with open(jc_entry.upstream_output_file) as f:
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
                        predict_args['model_path'] = up_model_path
                        predict_args['model_name'] = up_model_name
                        print("{}: add upstream model '{}' in '{}' from upstream output file '{}'"
                              .format(self.job_name, up_model_name, up_model_path, jc_entry.upstream_output_file))
                        break
        else:
            raise RuntimeError("both 'model_path' and upstream_output_file are not valid")

        runner = TFModelRunner(user_py_file, jc_entry.export_path, jc_entry.pack_path, predict_args=predict_args,
                               predict_data_args=predict_data_args)
        runner.run_predict()


if __name__ == '__main__':
    predictor = TFModelOfflinePredictor("TF model offline predictor componet")
    predictor.run()
