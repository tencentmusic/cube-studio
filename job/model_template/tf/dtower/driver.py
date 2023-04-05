# coding=utf-8
# @Time     : 2021/2/8 20:23
# @Auther   : lionpeng@tencent.com
import json

from job.model_template.utils import ModelTemplateDriver
from concurrent.futures import ProcessPoolExecutor
import traceback

DEBUG = False


def export_proc(job_name, job, pack_path, export_path):
    from job.tf_model_offline_predict.main import TFModelOfflinePredictor
    predictor = TFModelOfflinePredictor(job_name, args=[
        "--job", json.dumps(job),
        "--pack-path", pack_path,
        "--export-path", export_path
    ])
    predictor.run()


class DTowerModelTemplateDriver(ModelTemplateDriver):
    def __init__(self, output_file=None, args=None, local=False):
        super(DTowerModelTemplateDriver, self).__init__("DTower model template" + ("(debuging)" if DEBUG else ""),
                                                        output_file, args, local)

    def on_parsed_job(self, job: dict, pack_path: str, data_path: str) -> dict:
        job = super(DTowerModelTemplateDriver, self).on_parsed_job(job, pack_path, data_path)
        job_detail = job['job_detail']
        is_pairwise = job_detail['model_args'].get('pairwise', False)
        if is_pairwise:
            train_data_args = job_detail.get('train_data_args')
            if train_data_args:
                train_data_args['fake_label'] = 1.

            test_data_args = job_detail.get('test_data_args')
            if test_data_args:
                test_data_args['fake_label'] = 1.

            val_data_args = job_detail.get('val_data_args')
            if val_data_args:
                val_data_args['fake_label'] = 1.
        return job

    def post_proc(self, jc_entry, train_output_file: str):
        import copy
        job = jc_entry.job

        futures = []

        user_predict_args = job['job_detail'].get("user_predict_args")
        user_pred_data_args = job['job_detail'].get("user_pred_data_args")
        ppool = ProcessPoolExecutor(2)
        if user_predict_args and user_pred_data_args:
            if not user_pred_data_args.get('model_input_config_file', '').strip():
                user_pred_data_args['model_input_config_file'] = job['job_detail'].get('model_input_config_file')

            user_predict_job = copy.deepcopy(job)
            user_predict_job['job_detail']['predict_args'] = user_predict_args
            user_predict_job['job_detail']['pred_data_args'] = user_pred_data_args
            try:
                futures.append(ppool.submit(export_proc, self.job_name + " user embedding predictor",
                                            user_predict_job, jc_entry.pack_path, jc_entry.export_path))
                print("{}: user embedding prediction process started".format(self.job_name))
            except Exception as e:
                print("{}: WARING: start user embedding prediction process failed: {}\n{}"
                      .format(self.job_name, e, traceback.format_exc()))

        item_predict_args = job['job_detail'].get("item_predict_args")
        item_pred_data_args = job['job_detail'].get("item_pred_data_args")
        if item_predict_args and item_pred_data_args:
            if not item_pred_data_args.get('model_input_config_file', '').strip():
                item_pred_data_args['model_input_config_file'] = job['job_detail'].get('model_input_config_file')

            item_predict_job = copy.deepcopy(job)
            item_predict_job['job_detail']['predict_args'] = item_predict_args
            item_predict_job['job_detail']['pred_data_args'] = item_pred_data_args
            try:
                futures.append(ppool.submit(export_proc, self.job_name + " item embedding predictor",
                                            item_predict_job, jc_entry.pack_path, jc_entry.export_path))
                print("{}: item embedding prediction process started".format(self.job_name))
            except Exception as e:
                print("{}: WARING: start item embedding prediction process failed: {}\n{}"
                      .format(self.job_name, e, traceback.format_exc()))

        if futures:
            for f in futures:
                f.result()
            ppool.shutdown(True)


if __name__ == "__main__":
    if DEBUG:
        # import tensorflow as tf
        # tf.config.run_functions_eagerly(True)
        with open('./job_config_demo.json', 'r') as jcf:
            demo_job = json.load(jcf)
            driver = DTowerModelTemplateDriver(args=[
                "--job", json.dumps(demo_job),
                "--export-path", "./runs"
            ], local=True)
            driver.run()
    else:
        driver = DTowerModelTemplateDriver()
        driver.run()
