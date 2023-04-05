# coding=utf-8
# @Time     : 2021/1/13 20:59
# @Auther   : lionpeng@tencent.com

import re
from job.model_template.tf.youtube_dnn.custom_objects import *
from job.model_template.tf.youtube_dnn.export import export_user_embeddings, export_item_embeddings
from job.model_template.utils import ModelTemplateDriver
from job.pkgs.utils import make_abs_or_data_path

DEBUG = False


class YoutubeDNNTemplateDriver(ModelTemplateDriver):
    def __init__(self, output_file=None, args=None, local=False):
        super(YoutubeDNNTemplateDriver, self).__init__("YoutubeDNN model template" + ("(debuging)" if DEBUG else ""),
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
        train_data_args = job_detail.get('train_data_args', {})
        train_data_file = train_data_args.get('data_file', '').strip()
        train_file_column_names = None
        headers = train_data_args.get('headers')
        if headers:
            if isinstance(headers, str):
                headers = headers.split(',')
            train_file_column_names = list(filter(lambda x: len(x) > 0, map(lambda x: str(x).strip(), headers)))
        train_file_field_delim = train_data_args.get('field_delim')

        predict_data_file = job_detail.get('predict_data_file', '').strip() or train_data_file
        predict_batch_size = job_detail.get('predict_batch_size', 4096)
        write_buf_len = job_detail.get('write_buf_len')

        mi_cfg = ModelInputConfig.parse(job_detail.get('model_input_config_file'), jc_entry.pack_path,
                                        jc_entry.export_path)
        item_embedding_file = job_detail.get('item_embedding_file', '').strip()
        if item_embedding_file and train_data_file:
            item_embedding_fmt = job_detail.get('item_embedding_fmt', 'ef').strip()
            item_ts_bn = None
            item_emb_type = item_embedding_fmt[:2].lower()
            if item_emb_type == 'ts':
                m = re.match(r'^ts:([^:]+)$', item_embedding_fmt)
                if m is None:
                    raise RuntimeError("Terasearch embedding format must be specified as 'ts:<BN>', got '{}'"
                                       .format(item_embedding_fmt))
                item_ts_bn = m.group(1)
            elif item_embedding_fmt.lower() != 'ef':
                item_emb_type = None
                print("{}: WARNING: unrecognized 'item_embedding_fmt' '{}'".format(self.job_name, item_embedding_fmt))

            item_list_col_name = job_detail.get('item_list_col_name')
            if isinstance(item_list_col_name, str) and item_list_col_name.strip():
                item_list_col_name = [item_list_col_name.strip()]
            elif isinstance(item_list_col_name, (tuple, list)):
                item_list_col_name = list(filter(lambda x: x, map(lambda x: x.strip(), item_list_col_name)))

            if item_list_col_name:
                item_list_col_descs = [d for d in mi_cfg.all_inputs if d.name in item_list_col_name]
            else:
                item_list_col_descs = None
            item_col_desc = [d for d in mi_cfg.all_inputs if d.is_label]
            if item_col_desc:
                item_col_desc = item_col_desc[0]
            else:
                item_col_desc = None
            parallel = job_detail.get('item_embedding_predict_parallel', 1)
            export_item_embeddings(model_path, train_data_file, item_list_col_descs, item_col_desc,
                                   item_embedding_file, predict_batch_size, write_buf_len, parallel,
                                   item_emb_type, item_ts_bn, train_file_column_names, train_file_field_delim)
        else:
            print("'item_embedding_file' not set, will not saving item embeddings")

        user_embedding_file = job_detail.get('user_embedding_file', '').strip()
        if user_embedding_file and predict_data_file:
            uid_col_name = job_detail.get('uid_col_name', '').strip()
            if uid_col_name:
                user_embedding_fmt = job_detail.get('user_embedding_fmt', '').strip()
                user_ts_bn = None
                user_emb_type = user_embedding_fmt[:2].lower()
                if user_emb_type == 'ts':
                    m = re.match(r'^ts:([^:]+)$', user_embedding_fmt)
                    if m is None:
                        raise RuntimeError("Terasearch embedding format must be specified as 'ts:<BN>', got '{}'"
                                           .format(user_embedding_fmt))
                    user_ts_bn = m.group(1)
                elif user_embedding_fmt.lower() != 'ef':
                    user_emb_type = None
                    print("{}: WARNING: unrecognized 'user_embedding_fmt' '{}'"
                          .format(self.job_name, user_embedding_fmt))
                uid_col_desc = [d for d in mi_cfg.all_inputs if d.name == uid_col_name]
                if not uid_col_desc:
                    raise RuntimeError("column '{}' set in 'uid_col_name' not exists in model inputs"
                                       .format(uid_col_name))
                uid_col_desc = uid_col_desc[0]
                parallel = job_detail.get('user_embedding_predict_parallel', 1)
                export_user_embeddings(model_path, predict_data_file, mi_cfg.get_inputs_by_group('user'), uid_col_desc,
                                       user_embedding_file, predict_batch_size, write_buf_len, parallel, user_emb_type,
                                       user_ts_bn)
            else:
                print("'uid_col_name' not set, will not saving user embeddings")
        else:
            print("'user_embedding_file' not set, will not saving user embeddings")


if __name__ == "__main__":
    if DEBUG:
        # import tensorflow as tf
        # tf.config.run_functions_eagerly(True)
        with open('./job_config_demo.json', 'r') as jcf:
            demo_job = json.load(jcf)
            driver = YoutubeDNNTemplateDriver(args=[
                "--job", json.dumps(demo_job),
                "--export-path", "./runs"
            ], local=True)
            driver.run()
    else:
        driver = YoutubeDNNTemplateDriver()
        driver.run()
