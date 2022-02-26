
import inspect
import os
import queue
import re
import threading
import time
import sys
from datetime import datetime, timedelta
from dateutil import relativedelta

from .constants import PipelineParam


def parse_best_parameter(best_param_path: str, job: dict):
    import json
    best_params = json.load(open(best_param_path, 'r+'))

    TRAIN_ARGS_PARAMETERS = {
        'batch_size': None,
        'epochs': None,
        'train_type': None,
        'num_samples': None,
        'num_val_samples': None,
        'optimizer': None,
        'losses': None,
        'metrics': None,
        'early_stopping': None,
    }

    job_detail = job['job_detail']
    job_custom = True if ('params' in job_detail or 'model_args' not in job_detail) else False

    if job_custom:
        params = job['job_detail']['params']
        params_record = dict()
        if params is not None:
            for i,arg in enumerate(params):
                if i%2==0:
                    params_record[arg] = i
                    val = best_params.get(arg[2:])
                    if val is not None:
                        job['job_detail']['params'][i+1] = val
        for param in best_params:
            param_t = '--'+param if param[:2]!='--' else param
            if param_t not in params_record:
                if param_t in job['job_detail'].keys():
                    job['job_detail'][param_t] = best_params.get(param)
                else:
                    job['job_detail']['params'] += [param_t, best_params.get(param)]
    else:
        for train_arg in job['job_detail']['train_args'].keys():
            val = best_params.get(train_arg)
            if val is not None:
                job['job_detail']['train_args'][train_arg] = val
        for arg, val in best_params.items():
            if arg not in TRAIN_ARGS_PARAMETERS.keys():
                job['job_detail']['model_args'][arg] = val
    
    return job


def dynamic_load_class(pkg_path, class_name):
    add_path = os.path.dirname(os.path.realpath(pkg_path))
    sys.path.append(add_path)
    module_name = os.path.basename(pkg_path).split('.')[0]
    module = __import__(module_name)
    clazz = getattr(module, class_name)
    return clazz

def parse_size(size, int_result=True):
    if isinstance(size, (int, float)):
        return int(size) if int_result else float(size)
    if size is None:
        return 0 if int_result else 0.

    assert isinstance(size, str)
    size = size.strip()
    if not size:
        return 0 if int_result else 0.
    size = size.lower()
    m = re.match(r'^(\d+[.]?\d*)(k|m|g)?$', size)
    if not m:
        return None
    number = m.group(1)
    unit = m.group(2)
    number = float(number)
    if unit == 'k':
        number *= 2**10
    elif unit == 'm':
        number *= 2**20
    elif unit == 'g':
        number *= 2**30

    return int(number) if int_result else number


def parse_timedelta(delta_str):
    assert not delta_str or isinstance(delta_str, str)

    import datetime
    if not delta_str or not delta_str.strip():
        return None
    m = re.match(r'^(\d+)(ms|s|m|h|d|w)$', delta_str)
    if not m:
        return None
    number = int(m.group(1))
    unit = m.group(2)
    if unit == 'ms':
        return datetime.timedelta(milliseconds=number)
    elif unit == 's':
        return datetime.timedelta(seconds=number)
    elif unit == 'm':
        return datetime.timedelta(minutes=number)
    elif unit == 'h':
        return datetime.timedelta(hours=number)
    elif unit == 'd':
        return datetime.timedelta(days=number)
    elif unit == 'w':
        return datetime.timedelta(weeks=number)
    return None
    

def expand_path(path, run_path=None, pack_path=None, ignore_abs_path=True):
    if not path or not path.strip():
        return path
    path = path.strip()
    if ignore_abs_path and os.path.isabs(path):
        return path
    if run_path:
        run_path = run_path.strip()
    if pack_path:
        pack_path = pack_path.strip()
    if not run_path and not pack_path:
        return path

    if run_path:
        path = re.sub(PipelineParam.RUN_PATH_PAT, run_path, path)
    if pack_path:
        path = re.sub(PipelineParam.PACK_PATH_PAT, pack_path, path)
    return path


def make_abs_or_data_path(path, data_path, pack_path):
    if not path or not path.strip():
        return data_path
    path = expand_path(path, run_path=data_path, pack_path=pack_path)
    if os.path.isabs(path):
        return path
    return os.path.join(data_path, path)


def make_abs_or_pack_path(path, data_path, pack_path):
    if not path or not path.strip():
        return pack_path
    path = expand_path(path, run_path=data_path, pack_path=pack_path)
    if os.path.isabs(path):
        return path
    return os.path.join(pack_path, path)


def expand_param(param_val, data_path, pack_path):
    if not isinstance(param_val, str):
        return param_val

    if data_path:
        param_val = re.sub(PipelineParam.RUN_PATH_PAT, data_path, param_val)
    if pack_path:
        param_val = re.sub(PipelineParam.PACK_PATH_PAT, pack_path, param_val)

    expanded = param_val
    for m in re.finditer(PipelineParam.DATE_PAT, param_val):
        date = datetime.now()
        f, n, unit, fmt = m.group(2, 3, 4, 6)
        if all([f, n, unit]):
            delta_num = int(f+n)
            if unit == 'd':
                date = date + timedelta(days=delta_num)
            elif unit == 'w':
                date = date + timedelta(weeks=delta_num)
            elif unit == 'h':
                date = date + timedelta(hours=delta_num)
            elif unit == 'm':
                date = date + timedelta(minutes=delta_num)
            elif unit == 's':
                date = date + timedelta(seconds=delta_num)
            elif unit == 'M':
                date = date + relativedelta.relativedelta(months=delta_num)
            elif unit == 'y':
                date = date + relativedelta.relativedelta(years=delta_num)

        if not fmt:
            fmt = "%Y%m%d"
        expanded = expanded.replace(m.group(0), date.strftime(fmt))
    return expanded


def recur_expand_param(param, data_path, pack_path):
    if not param:
        return param
    if isinstance(param, (list, tuple)):
        expaned = []
        for i in param:
            expaned.append(recur_expand_param(i, data_path, pack_path))
        if isinstance(param, tuple):
            return tuple(expaned)
        return expaned
    elif isinstance(param, dict):
        expanded = {}
        for k, v in param.items():
            expanded[k] = recur_expand_param(v, data_path, pack_path)
        return expanded
    else:
        return expand_param(param, data_path, pack_path)


def split_file_name(file_name):
    if not file_name:
        return '', '', ''
    dir_name = os.path.dirname(file_name)
    name = os.path.basename(file_name)
    base, ext = os.path.splitext(name)
    return dir_name, base, ext


def try_archive_by_config(config_json, data_path, pack_path):
    if not config_json:
        return None
    if not isinstance(config_json, list):
        config_json = [config_json]
    from .archiver import Archiver
    archiver = Archiver()
    archived = []
    for i, cj in enumerate(config_json):
        src = cj.get('src', '').strip()
        if not src:
            print("'src' of {}th archive not set, ignore it".format(i))
            continue
        src = make_abs_or_data_path(src, data_path, pack_path)
        path_name = cj.get('path_name', '').strip()
        compress = cj.get('compress', False)
        cj_archived = archiver.archive(src, path_name, compress)
        if cj_archived:
            archived.extend(cj_archived)
    return archived


def call_user_module(module, func_name, func_must_exists, nullable, check_return_type,
                     inject_args: dict = None, **kwargs):
    injected_args = {}
    if not hasattr(module, func_name):
        if func_must_exists:
            raise ModuleNotFoundError("user function '{}' not found in module {}".format(func_name, module))
        else:
            return None, None

    func = getattr(module, func_name)
    args_spec = inspect.getfullargspec(func)
    varkw_args = {}
    if inject_args:
        for a, v in inject_args.items():
            if a in args_spec.args:
                kwargs[a] = v
                injected_args[a] = v
                print("user function '{}' of module {} declared arg '{}', inject value '{}'"
                      .format(func_name, module, a, v))
            elif args_spec.varkw:
                varkw_args[a] = v
                injected_args[a] = v
                print("user function '{}' of module {} declared kw arg '{}', inject '{}'={} into it"
                      .format(func_name, module, args_spec.varkw, a, v))

    not_support_args = kwargs.keys() - args_spec.args
    if not_support_args:
        for nsa in not_support_args:
            v = kwargs.pop(nsa)
            if args_spec.varkw:
                if nsa not in varkw_args:
                    varkw_args[nsa] = v
                    print("'{}'={} in kwargs not decleared in use function '{}' of module {},"
                          " moved it into kw arg '{}'".format(nsa, v, func_name, module, args_spec.varkw))
            else:
                print("'{}'={} in kwargs not decleared in use function '{}' of module {}, will be excluded"
                      .format(nsa, v, func_name, module))
    ret_obj = func(**kwargs, **varkw_args)
    if ret_obj is None and not nullable:
        raise RuntimeError("user function '{}' of module {} return None, args={}"
                           .format(func_name, module, kwargs))

    if ret_obj is not None and check_return_type is not None and not isinstance(ret_obj, check_return_type):
        raise RuntimeError("object '{}' returned by user function '{}' of module {} is of not type '{}'"
                           .format(ret_obj, func_name, module, check_return_type))

    return ret_obj, injected_args


def find_duplicated_entries(seq):
    if not seq:
        return None
    from collections import Counter
    cnt = Counter(seq)
    duplicated_entries = list(map(lambda i: i[0], filter(lambda x: x[1] > 1, cnt.items())))
    return duplicated_entries


class _WriterThread(threading.Thread):
    def __init__(self, q: queue.Queue, f, max_batch_len, name='bufferd_text_file_writer_thread'):
        super(_WriterThread, self).__init__()
        self.q = q
        self.f = f
        self.setName(name)
        self.wrote_lines = 0
        self.wrote_times = 0
        self.wrote_cost_time = 0
        self.max_batch_len = max_batch_len
        self.batch = []
        self.first_write_time = None

    def _flush(self):
        if not self.batch:
            return
        num_lines = len(self.batch)

        st = time.perf_counter()
        if self.first_write_time is None:
            self.first_write_time = st
        data = ''.join(self.batch)
        self.f.write(data)
        self.wrote_lines += num_lines
        self.wrote_times += 1
        cost = time.perf_counter() - st
        self.wrote_cost_time += cost
        self.batch.clear()
        print("{}: wrote {} lines, cost {}s, totally write {} times with {} lines, cost {}s, elapsed {}s"
              .format(self.getName(), num_lines, cost, self.wrote_times, self.wrote_lines,
                      self.wrote_cost_time, time.perf_counter()-self.first_write_time))

    def run(self) -> None:
        if not self.q:
            raise RuntimeError("{}: no queue specified".format(self.getName()))
        print("{}: started".format(self.getName()))
        while True:
            try:
                if len(self.batch) >= self.max_batch_len/2:
                    line = self.q.get_nowait()
                else:
                    line = self.q.get()
                if line == BufferedTextFileWriter.END_MARK:
                    self._flush()
                    print("{}: received end mark, exit, totally write {} times with {} lines, cost {}s, elapsed {}s"
                          .format(self.getName(), self.wrote_times, self.wrote_lines, self.wrote_cost_time,
                                  time.perf_counter()-self.first_write_time))
                    return
                self.batch.append(line)
                if len(self.batch) >= self.max_batch_len:
                    self._flush()
            except queue.Empty:
                self._flush()


class BufferedTextFileWriter(object):
    END_MARK = '__END_MARK__'

    def __init__(self, filename, line_buffer_len, sys_buffer_size=2**25):
        self.filename = filename
        self.line_buffer_len = line_buffer_len
        self.sys_buffer_size = sys_buffer_size
        self.opened_fn = None
        self.q = None
        self._write_thread = None

    def __enter__(self):
        if self.opened_fn is None:
            self.opened_fn = open(self.filename, 'w', buffering=self.sys_buffer_size)
            self.q = queue.Queue(self.line_buffer_len)
            self._write_thread = _WriterThread(self.q, self.opened_fn, self.line_buffer_len,
                                               "writerthread: {}".format(self.filename))
            self._write_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.opened_fn is not None:
            self.q.put(self.END_MARK)
            print("begin waiting writer thread to terminate")
            self._write_thread.join()
            print("writer thread terminated")
            self.opened_fn.close()
            self.opened_fn = None
            self.q = None
            self._write_thread = None

    def write(self, line):
        if self.opened_fn is None:
            raise RuntimeError("please use with syntax to use BufferedTextFileWriter")
        if not line:
            return 0
        self.q.put(line)
        return 1

    @property
    def wrote_lines(self):
        if self._write_thread is None:
            return 0
        return self._write_thread.wrote_lines


def even_spread_num(n, num_buckets):
    d, r = divmod(n, num_buckets)
    buckets = [d]*num_buckets
    for i in range(r):
        buckets[i] += 1
    return buckets


def find_files(file_patterns):
    if not isinstance(file_patterns, (list, tuple, str)):
        print("WARNING: file_patterns should be list/tuple/str, got '{}': {}"
              .format(type(file_patterns), file_patterns))
        return []
    files = []
    if not isinstance(file_patterns, (list, tuple)):
        file_patterns = [file_patterns]

    import glob
    for fp in file_patterns:
        files.extend(glob.glob(fp))

    return files


def get_file_md5(file_path):
    import hashlib
    if not os.path.isfile(file_path):
        return None
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while True:
            d = f.read(4 << 20)
            if not d:
                break
            md5.update(d)
    return str(md5.hexdigest())
