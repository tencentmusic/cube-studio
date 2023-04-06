'''
Author: your name
Date: 2021-06-09 17:07:22
LastEditTime: 2021-06-25 14:51:21
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /job-template/job/model_template/tf/ple/driver.py
'''
import json
import os

from job.model_template.utils import ModelTemplateDriver

DEBUG = False


class PLEModelTemplateDriver(ModelTemplateDriver):
    def __init__(self, output_file=None, args=None, local=False):
        super(PLEModelTemplateDriver, self).__init__("PLE model template" + ("(debuging)" if DEBUG else ""),
                                                      output_file, args, local)


if __name__ == "__main__":
    if DEBUG:
        # import tensorflow as tf
        # tf.config.run_functions_eagerly(True)
        with open('/Users/jurluo/tme-kubeflow/job-template/job/model_template/tf/ple/config_template_debug.json', 'r') as jcf:
            demo_job = json.load(jcf)
            driver = PLEModelTemplateDriver(args=[
                "--job", json.dumps(demo_job),
                "--export-path", '/Users/jurluo/tme-kubeflow/job-template/job/model_template/tf/ple/runs',
                "--pack-path", '/Users/jurluo/tme-kubeflow/job-template/job/model_template/tf/ple'
            ], local=True)
            driver.run()
    else:
        driver = PLEModelTemplateDriver()
        driver.run()
