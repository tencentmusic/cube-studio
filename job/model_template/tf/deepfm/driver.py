# coding=utf-8
# @Time     : 2021/2/8 20:23
# @Auther   : lionpeng@tencent.com
import json

import tensorflow as tf
from job.model_template.utils import ModelTemplateDriver

DEBUG = False


class DeeFMTemplateDriver(ModelTemplateDriver):
    def __init__(self, output_file=None, args=None, local=False):
        super(DeeFMTemplateDriver, self).__init__("DeepFM model template" + ("(debuging)" if DEBUG else ""),
                                                  output_file, args, local)


if __name__ == "__main__":
    if DEBUG:
        # import tensorflow as tf
        # tf.config.run_functions_eagerly(True)
        # tf.compat.v1.disable_eager_execution()
        with open('./job_config_demo.json', 'r') as jcf:
            demo_job = json.load(jcf)
            driver = DeeFMTemplateDriver(args=[
                "--job", json.dumps(demo_job),
                "--export-path", "./runs"
            ], local=True)
            driver.run()
    else:
        driver = DeeFMTemplateDriver()
        driver.run()
