import json

from job.model_template.utils import ModelTemplateDriver
from job.model_template.tf.comirec.custom_objects import *
from job.model_template.tf.comirec.export import export_user_embeddings, export_item_embeddings
from job.pkgs.utils import make_abs_or_data_path

DEBUG = False


class ComiRecModelTemplateDriver(ModelTemplateDriver):
    def __init__(self, output_file=None, args=None, local=False):
        super(ComiRecModelTemplateDriver, self).__init__("ComiRec model template" + ("(debuging)" if DEBUG else ""),
                                                      output_file, args, local)

    def post_proc(self, jc_entry, train_output_file: str):
        train_output_file = make_abs_or_data_path(train_output_file, jc_entry.data_path, jc_entry.pack_path)
        if not os.path.isfile(train_output_file):
            print("{}: WARNING: training output file '{}' not exists, can not export embeddings"
                  .format(self.job_name, train_output_file))
            return

        model_path = None
        with open(train_output_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                model_path = line.split('|')[0]
                break
        if not model_path or not os.path.exists(model_path):
            print("{}: WARNING: model path '{}' not exists, can not export embeddings"
                  .format(self.job_name, model_path))
            return

        job_detail = jc_entry.job['job_detail']
        train_data_file = job_detail.get('train_data_args', {}).get('data_file', '').strip()
        predict_data_file = job_detail.get('predict_data_file', '').strip() or train_data_file
        predict_batch_size = job_detail.get('predict_batch_size', 4096)
        write_buf_len = job_detail.get('write_buf_len')

        mi_cfg = ModelInputConfig.parse(job_detail.get('model_input_config_file'), jc_entry.pack_path,
                                        jc_entry.export_path)
        item_embedding_file = job_detail.get('item_embedding_file', '').strip()
        if item_embedding_file and train_data_file:
            item_list_col_name = job_detail.get('item_list_col_name')
            if isinstance(item_list_col_name, str) and item_list_col_name.strip():
                item_list_col_name = [item_list_col_name.strip()]
            elif isinstance(item_list_col_name, (tuple, list)):
                item_list_col_name = list(filter(lambda x: x, map(lambda x: x.strip(), item_list_col_name)))

            if item_list_col_name:
                item_list_col_descs = [d for d in mi_cfg.all_inputs if d.name in item_list_col_name] # other item_id lists such as history_watch, history_like...
            else:
                item_list_col_descs = None
            item_col_desc = [d for d in mi_cfg.all_inputs if d.is_label] # label includes item_id
            if item_col_desc:
                item_col_desc = item_col_desc[0]
            else:
                item_col_desc = None
            parallel = job_detail.get('item_embedding_predict_parallel', 1)
            export_item_embeddings(model_path, train_data_file, item_list_col_descs, item_col_desc,
                                   item_embedding_file, predict_batch_size, write_buf_len, parallel)
        else:
            print("'item_embedding_file' not set, will not saving item embeddings")

        user_embedding_file = job_detail.get('user_embedding_file', '').strip()
        if user_embedding_file and predict_data_file:
            uid_col_name = job_detail.get('uid_col_name', '').strip()
            if uid_col_name:
                uid_col_desc = [d for d in mi_cfg.all_inputs if d.name == uid_col_name]
                if not uid_col_desc:
                    raise RuntimeError("column '{}' set in 'uid_col_name' not exists in model inputs"
                                       .format(uid_col_name))
                uid_col_desc = uid_col_desc[0]
                parallel = job_detail.get('user_embedding_predict_parallel', 1)
                user_item_input_descs = mi_cfg.get_inputs_by_group('user')+mi_cfg.get_inputs_by_group('item')
                export_user_embeddings(model_path, predict_data_file, user_item_input_descs, uid_col_desc,
                                       user_embedding_file, predict_batch_size, write_buf_len, parallel)
            else:
                print("'uid_col_name' not set, will not saving user embeddings")
        else:
            print("'user_embedding_file' not set, will not saving user embeddings")


if __name__ == "__main__":
    if DEBUG:
        # import tensorflow as tf
        # tf.config.run_functions_eagerly(True)

        import argparse
        parser = argparse.ArgumentParser(description="ComiRecmodel arguments")
        # parser.add_argument('--use_gpu', help='if use gpu during training and evaluating', default=False, action='store_true')
        # parser.add_argument('--config_input_path', type=str, help='config input path', default='/data/ft_local/dataset_movielens/movielens_input_config.json')
        # parser.add_argument('--config_template_path', type=str, help='config template path', default='/data/ft_local/dataset_movielens/comirec_template_config_movielens.json')
        # parser.add_argument('--export_path', type=str, help='export path', default='/data/ft_local/job-runs/runs-movielens-comirec_n3/')
        # parser.add_argument('--config_input_path', type=str, help='config input path', default='/data/ft_local/dataset_book/amazonbook_input_config.json')
        # parser.add_argument('--config_template_path', type=str, help='config template path', default='/data/ft_local/dataset_book/comirec_template_config_book.json')
        # parser.add_argument('--export_path', type=str, help='export path', default='/data/ft_local/job-runs/runs-book-comirec2/')
        # args = parser.parse_args()
        with open("./job_config_demo.json", 'r') as jcf:
            # if args.use_gpu==False:
            #     import os
            #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
            #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
            demo_job = json.load(jcf)
            # demo_job['job_detail']['model_input_config_file'] = args.config_input_path
            driver = ComiRecModelTemplateDriver(args=[
                "--job", json.dumps(demo_job),
                "--export-path", "/mnt/data/test_datas/comirec",
                "--pack-path", "./"
            ], local=True)
            driver.run()

        # with open('./job_config_demo.json', 'r') as jcf:
        #     demo_job = json.load(jcf)
        #     driver = ComiRecModelTemplateDriver(args=[
        #         "--job", json.dumps(demo_job),
        #         "--export-path", "./runs"
        #     ], local=True)
        #     driver.run()
    else:
        driver = ComiRecModelTemplateDriver()
        driver.run()
