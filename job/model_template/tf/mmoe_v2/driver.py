import json

from job.model_template.utils import ModelTemplateDriver

DEBUG = False


class MMOEModelTemplateDriver(ModelTemplateDriver):
    def __init__(self, output_file=None, args=None, local=False):
        super(MMOEModelTemplateDriver, self).__init__("MMoE V2 model template" + ("(debuging)" if DEBUG else ""),
                                                      output_file, args, local)

if __name__ == "__main__":
    if DEBUG:
        import argparse
        parser = argparse.ArgumentParser(description="mmoe v2 model arguments")
        parser.add_argument('--use_gpu', help='if use gpu during training and evaluating', default=False, action='store_true')
        parser.add_argument('--config_input_path', type=str, help='config input path', default='/root/ft_local/dataset_qq_music/mmoe_qqmusic.json')
        parser.add_argument('--config_template_path', type=str, help='config template path', default='/root/ft_local/dataset_qq_music/config_template_qqmusic_mmoev2_2.json')
        parser.add_argument('--export_path', type=str, help='export path', default='/root/ft_local/job-runs/runs-test2/')
        args = parser.parse_args()
        with open(args.config_template_path, 'r') as jcf:
            if args.use_gpu==False:
                import os
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
            demo_job = json.load(jcf)
            demo_job['job_detail']['model_input_config_file'] = args.config_input_path
            driver = MMOEModelTemplateDriver(args=[
                "--job", json.dumps(demo_job),
                "--export-path", args.export_path
            ], local=True)
            driver.run()

        # with open('./config_template_small.json', 'r') as jcf:
        #     # import os
        #     # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
        #     # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
        #     demo_job = json.load(jcf)
        #     driver = MMOEModelTemplateDriver(args=[
        #         "--job", json.dumps(demo_job),
        #         "--export-path", "/root/ft_local/runs-small-cpu/"
        #     ], local=True)
        #     driver.run()
    else:
        driver = MMOEModelTemplateDriver()
        driver.run()
