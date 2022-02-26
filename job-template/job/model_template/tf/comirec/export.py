

from job.pkgs.tf.feature_util import *
from job.pkgs.utils import BufferedTextFileWriter, even_spread_num
from job.model_template.tf.comirec.custom_objects import custom_objects
from job.model_template.tf.data_helper import extract_csv_input_header
from job.pkgs.utils import split_file_name
from typing import List
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
import glob
from math import ceil


class ItemEmbedddingExportProcess(object):
    def __init__(self, idx):
        self.idx = idx
        self.name = "item-emb-export-proc-{}".format(self.idx)

    def __call__(self, model_path, predict_data_files, item_list_col_descs: List[InputDesc],
                 item_col_desc: InputDesc, vec_file, predict_batch_size=4096,
                 write_buf_len=None):
        print("{}: started, predict_data_files={}, vec_file={}".format(self.name, predict_data_files, vec_file))
        if not predict_data_files:
            print("{}: predict_data_files not specifed: {}".format(self.name, predict_data_files))
            return 0
        try:
            model = tf.keras.models.load_model(model_path, custom_objects, compile=False)
            print("{}: loaded model '{}' from path '{}'".format(self.name, model, model_path))
        except Exception as e:
            print("{}: failed to load model from path '{}': {}\n{}"
                  .format(self.name, model_path, e, traceback.format_exc()))
            raise e

        item_list_ds = None
        if item_list_col_descs is not None:
            if not isinstance(item_list_col_descs, (list, tuple)):
                item_list_col_descs = [item_list_col_descs]
            if item_list_col_descs:
                def __split_item_list(item_list, col_desc: InputDesc):
                    items = tf.strings.split(item_list[col_desc.name], col_desc.val_sep)
                    items = tf.strings.to_number(items, out_type=col_desc.dtype)
                    return items

                item_list_ds = None
                for item_list_col_desc in item_list_col_descs:
                    ds = tf.data.experimental.make_csv_dataset(predict_data_files,
                                                               batch_size=1024, num_epochs=1, field_delim=' ',
                                                               num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                                               prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
                                                               select_columns=[item_list_col_desc.name],
                                                               column_defaults=[
                                                                   item_list_col_desc.infer_default_value()],
                                                               shuffle=False)

                    ds = ds.map(lambda x: __split_item_list(x, item_list_col_desc), tf.data.experimental.AUTOTUNE) \
                        .unbatch().flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
                    if item_list_ds is None:
                        item_list_ds = ds
                    else:
                        item_list_ds = item_list_ds.concatenate(ds).apply(tf.data.experimental.unique())

        item_ds = None
        if item_col_desc is not None:
            item_ds = tf.data.experimental.make_csv_dataset(predict_data_files,
                                                            batch_size=1024, num_epochs=1, field_delim=' ',
                                                            num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                                            prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
                                                            select_columns=[item_col_desc.name],
                                                            column_defaults=[item_col_desc.infer_default_value()],
                                                            shuffle=False)
            item_ds = item_ds.map(lambda x: tf.cast(x[item_col_desc.name], item_col_desc.dtype)).unbatch() \
                .apply(tf.data.experimental.unique())

        if item_list_ds is not None and item_ds is not None:
            all_item_ds = item_list_ds.concatenate(item_ds)
        elif item_list_ds is not None:
            all_item_ds = item_list_ds
        elif item_ds is not None:
            all_item_ds = item_ds
        else:
            print("{}: WARNING: no item embeddings to write".format(self.name))
            return 0

        all_item_ds = all_item_ds.apply(tf.data.experimental.unique()).batch(predict_batch_size)
        print("{}: begin writing item embeddings to '{}'".format(self.name, vec_file))
        write_cnt = 0
        st = time.perf_counter()
        if write_buf_len is None or write_buf_len <= 0:
            write_buf_len = predict_batch_size * 2
            print("{}: write_buf_len={}".format(self.name, write_buf_len))
        # with BufferedTextFileWriter(vec_file, write_buf_len) as f:
        with open(vec_file, 'w', buffering=2 ** 25) as f:
            for item_ids in all_item_ds:
                pred_st = time.perf_counter()
                embeddings = model.get_item_feature_embeddings(tf.reshape(item_ids, (-1, 1)), item_col_desc.name)
                print("{}: predicted {} item embeddings, cost {}s".format(self.name, item_ids.shape[0],
                                                                          time.perf_counter()-pred_st))
                for item_id, embedding in zip(item_ids, embeddings):
                    item_id = item_id.numpy()
                    embedding_str = ','.join(map(str, embedding.numpy()))
                    line = '{' + \
                           '"MD": "{}","BN": "push_playlist_recall", "VA": "txt|{}"'.format(item_id, embedding_str)\
                           + '}'
                    f.write(line + '\n')
                    write_cnt += 1
                    if write_cnt > 0 and write_cnt % write_buf_len == 0:
                        print("{}: wrote {} item embeddings, cost {}s".format(self.name, write_cnt,
                                                                              time.perf_counter()-st))

        print("{}: write item embeddings to '{}' finished, totaly {} items, cost {}s"
              .format(self.name, vec_file, write_cnt, time.perf_counter()-st))
        del model
        print("{}: released loaded model from '{}'".format(self.name, model_path))
        return write_cnt


class UserEmbeddingExportProcess(object):
    def __init__(self, idx):
        self.idx = idx
        self.name = "user-emb-export-proc-{}".format(self.idx)

    def __call__(self, model_path, predict_data_files, user_input_descs: List[InputDesc],
                 uid_col_desc: InputDesc, vec_file, predict_batch_size=4096,
                 write_buf_len=None):
        print("{}: started, predict_data_files={}, vec_file={}".format(self.name, predict_data_files, vec_file))
        if not predict_data_files:
            print("{}: predict_data_files not specifed: {}".format(self.name, predict_data_files))
            return 0
        try:
            model = tf.keras.models.load_model(model_path, custom_objects, compile=False)
            print("{}: loaded model '{}' from path '{}'".format(self.name, model, model_path))
        except Exception as e:
            print("{}: failed to load model from path '{}': {}\n{}"
                  .format(self.name, self.model_path, e, traceback.format_exc()))
            raise e

        if not isinstance(predict_data_files, (list, tuple)):
            predict_data_files = [predict_data_files]
            print("{}: set predict_data_files={}".format(self.name, predict_data_files))
        files = list(chain.from_iterable([glob.glob(f) for f in predict_data_files]))
        if not files:
            print("{}: WARING: found no files from file patterns: {}".format(self.name, predict_data_files))

        all_cols, sel_col_indices, sel_col_defaults = extract_csv_input_header(
            user_input_descs + [uid_col_desc], files[0], ',')
        ds = tf.data.experimental.make_csv_dataset(predict_data_files,
                                                   batch_size=predict_batch_size, num_epochs=1, field_delim=' ',
                                                   num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                                   prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
                                                   select_columns=sel_col_indices, column_defaults=sel_col_defaults,
                                                   shuffle=False)

        print("{}: begin writing user embeddings to '{}'".format(self.name, vec_file))
        write_cnt = 0
        st = time.perf_counter()
        if write_buf_len is None or write_buf_len <= 0:
            write_buf_len = predict_batch_size * 2
            print("{}: write_buf_len={}".format(self.name, write_buf_len))
        # with BufferedTextFileWriter(vec_file, write_buf_len) as f:
        with open(vec_file, 'w', buffering=2**25) as f:
            # f.write("uin\tembedding\n")
            for batch in ds:
                uids = batch.get(uid_col_desc.name)
                pred_st = time.perf_counter()
                target_feats = model.get_target_inputs(batch)
                for key in batch.keys():
                    if key in target_feats:
                        batch.pop(key)
                for key,val in batch.items():
                    batch[key] = tf.reshape(val, [-1,1])
                embeddings = model.predict(batch).numpy()
                print("{}: predicted {} user embeddings, cost {}s".format(self.name, uids.shape[0],
                                                                          time.perf_counter()-pred_st))
                for uid, embedding in zip(uids, embeddings):
                    uid = uid.numpy()
                    embedding_str = ''
                    for interest in embedding:
                        embedding_str += '\t'
                        embedding_str += ','.join(map(str, interest))
                    line = "{}{}".format(uid, embedding_str)
                    f.write(line + '\n')
                    write_cnt += 1
                    if write_cnt > 0 and write_cnt % write_buf_len == 0:
                        print("{}: wrote {} user embeddings, cost {}s".format(self.name, write_cnt,
                                                                              time.perf_counter()-st))
        print("{}: write user embeddings to '{}' finished, totaly {} users, cost {}s"
              .format(self.name, vec_file, write_cnt, time.perf_counter()-st))
        del model
        print("{}: released loaded model from '{}'".format(self.name, model_path))
        return write_cnt


def __parallel_prepare(predict_data_files, vec_file, parallel):
    if not isinstance(predict_data_files, (list, tuple)):
        predict_data_files = [predict_data_files]
    src_files = list(chain.from_iterable([glob.glob(f) for f in predict_data_files]))

    if not src_files:
        print("found no predict data files for from patterns '{}'".format(predict_data_files))
        return None
    if parallel is None or parallel <= 0:
        parallel = len(src_files)
    else:
        parallel = min(parallel, len(src_files))

    print("parallel={}, src_files={}".format(parallel, src_files))
    src_tar_pairs = []
    if parallel == 1:
        src_tar_pairs.append((src_files, vec_file))
    else:
        per_worker_src_nums = even_spread_num(len(src_files), parallel)
        path, base, ext = split_file_name(vec_file)
        offset = 0
        for i, num in enumerate(per_worker_src_nums):
            srcs = src_files[offset:offset+num]
            offset += num
            tar = os.path.join(path, base+"-part-"+str(i)+ext)
            src_tar_pairs.append((srcs, tar))

    return src_tar_pairs


def export_item_embeddings(model_path, predict_data_files, item_list_col_descs: List[InputDesc],
                           item_col_desc: InputDesc, vec_file, predict_batch_size=4096,
                           write_buf_len=None, parallel=None):
    src_tar_pairs = __parallel_prepare(predict_data_files, vec_file, parallel)
    if not src_tar_pairs:
        return 0
    print("item src_tar_pairs({})={}".format(len(src_tar_pairs), src_tar_pairs))
    ppool = ProcessPoolExecutor(parallel)
    p_rets = []
    st = time.perf_counter()
    for i, (srcs, tar) in enumerate(src_tar_pairs):
        ret = ppool.submit(ItemEmbedddingExportProcess(i), model_path, srcs, item_list_col_descs,
                           item_col_desc, tar, predict_batch_size, write_buf_len)
        p_rets.append(ret)

    fi = 0
    total_cnt = 0
    for ret in as_completed(p_rets):
        cnt = ret.result()
        fi += 1
        total_cnt += cnt
        print("{}/{} item embedding export process completed, {} items, totally {} items, cost {}s"
              .format(fi, len(p_rets), cnt, total_cnt, time.perf_counter()-st))

    ppool.shutdown(wait=True)
    print("export item embeddings finished, cost {}s".format(time.perf_counter() - st))
    return total_cnt


def export_user_embeddings(model_path, predict_data_files, user_input_descs: List[InputDesc],
                           uid_col_desc: InputDesc, vec_file, predict_batch_size=4096,
                           write_buf_len=None, parallel=None):
    src_tar_pairs = __parallel_prepare(predict_data_files, vec_file, parallel)
    if not src_tar_pairs:
        return 0
    print("user src_tar_pairs({})={}".format(len(src_tar_pairs), src_tar_pairs))
    ppool = ProcessPoolExecutor(parallel)
    p_rets = []
    st = time.perf_counter()
    for i, (srcs, tar) in enumerate(src_tar_pairs):
        ret = ppool.submit(UserEmbeddingExportProcess(i), model_path, srcs, user_input_descs,
                           uid_col_desc, tar, predict_batch_size, write_buf_len)
        p_rets.append(ret)

    fi = 0
    total_cnt = 0
    for ret in as_completed(p_rets):
        cnt = ret.result()
        fi += 1
        total_cnt += cnt
        print("{}/{} user embedding export process completed, {} users, totally {} users, cost {}s"
              .format(fi, len(p_rets), cnt, total_cnt, time.perf_counter()-st))
    ppool.shutdown(wait=True)
    print("export user embeddings finished, cost {}s".format(time.perf_counter()-st))

    return total_cnt
