import json

from job.model_template.utils import ModelTemplateDriver

DEBUG = False


class DINModelTemplateDriver(ModelTemplateDriver):
    def __init__(self, output_file=None, args=None, local=False):
        super(DINModelTemplateDriver, self).__init__("DIN model template" + ("(debuging)" if DEBUG else ""),
                                                      output_file, args, local)


if __name__ == "__main__":
    if DEBUG:
        import argparse
        parser = argparse.ArgumentParser(description="DINModel arguments")
        parser.add_argument('--use_gpu', help='if use gpu during training and evaluating', default=False, action='store_true')
        # parser.add_argument('--config_input_path', type=str, help='config input path', default='/data/ft_local/dataset_movielens/movielens20m_input_config.json')
        # parser.add_argument('--config_template_path', type=str, help='config template path', default='/data/ft_local/dataset_movielens/din_template_config_movielens20m.json')
        # parser.add_argument('--export_path', type=str, help='export path', default='/data/ft_local/job-runs/runs-movielens20m_din10/')
        parser.add_argument('--config_input_path', type=str, help='config input path', default='/data/ft_local/dataset_electronic/amazonelectronic_input_config.json')
        parser.add_argument('--config_template_path', type=str, help='config template path', default='/data/ft_local/dataset_electronic/din_template_config_electronic.json')
        parser.add_argument('--export_path', type=str, help='export path', default='/data/ft_local/job-runs/runs-electronic_din2/')

        args = parser.parse_args()
        with open(args.config_template_path, 'r') as jcf:
            if args.use_gpu==False:
                import os
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
            demo_job = json.load(jcf)
            demo_job['job_detail']['model_input_config_file'] = args.config_input_path
            driver = DINModelTemplateDriver(args=[
                "--job", json.dumps(demo_job),
                "--export-path", args.export_path
            ], local=True)
            driver.run()

        # import tensorflow as tf
        # tf.config.run_functions_eagerly(True)
        # with open('./job_config_demo.json', 'r') as jcf:
        #     demo_job = json.load(jcf)
        #     driver = DINModelTemplateDriver(args=[
        #         "--job", json.dumps(demo_job),
        #         "--export-path", "./runs"
        #     ], local=True)
        #     driver.run()
    else:
        driver = DINModelTemplateDriver()
        driver.run()
