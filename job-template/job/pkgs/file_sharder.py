
import os
import csv
import typing
import queue
import time
from threading import Barrier
from concurrent.futures import ThreadPoolExecutor
from job.pkgs.utils import split_file_name


class CsvFileReader(object):
    def __init__(self, file_name, field_delim):
        self.file_name = file_name
        self.field_delim = field_delim
        self.opened_f = None
        self.reader = None

    def get_headers(self):
        with open(self.file_name, 'r') as f:
            reader = csv.reader(f, delimiter=self.field_delim)
            headers = list(next(reader))
        return headers

    def count_records(self):
        with open(self.file_name, 'r') as f:
            reader = csv.reader(f, delimiter=self.field_delim)
            row_count = sum(1 for r in reader)
        return row_count-1

    def __enter__(self):
        self.opened_f = open(self.file_name, 'r', encoding='utf8')
        self.reader = csv.reader(self.opened_f, delimiter=self.field_delim)
        return self.reader

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.opened_f is not None:
            self.opened_f.close()
            self.opened_f = None
            self.reader = None


class CsvFileWriter(object):
    def __init__(self, file_name, field_delim):
        self.file_name = file_name
        self.field_delim = field_delim
        self.opend_f = None
        self.writer = None

    def __enter__(self):
        self.opend_f = open(self.file_name, 'w', encoding='utf8')
        self.writer = csv.writer(self.opend_f, delimiter=self.field_delim)
        return self.writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.opend_f is not None:
            self.opend_f.close()
            self.opend_f = None
            self.writer = None


class CsvFileSharder(object):
    END_MARK = '__END_OF_QUEUE__'

    def __init__(self, source_file_patterns, result_file_count, tar_path=None, ret_file_name_prefix='',
                 field_delim=' ', write_buf_len=0, remove_empty_files=True, header=True):
        if isinstance(source_file_patterns, str) and source_file_patterns.strip():
            self.source_file_patterns = [source_file_patterns.strip()]
        elif isinstance(source_file_patterns, (tuple, list)):
            self.source_file_patterns = list(filter(lambda x: x, map(lambda x: x.strip(), source_file_patterns)))
        elif source_file_patterns is not None:
            raise RuntimeError("'source_file_patterns' should be a string or list/tuple of strings, got '{}': {}"
                               .format(type(source_file_patterns), source_file_patterns))

        if not isinstance(result_file_count, int) or result_file_count <= 0:
            raise RuntimeError("'result_file_count' should be positive int, got '{}': {}"
                               .format(type(result_file_count), result_file_count))

        if tar_path is not None and not isinstance(tar_path, str):
            raise RuntimeError("'tar_path' should be None/string, got '{}': {}".format(type(tar_path), tar_path))
        elif isinstance(tar_path, str):
            tar_path = tar_path.strip()
            if tar_path and not os.path.exists(tar_path):
                os.makedirs(tar_path, exist_ok=True)
                print("created shard file path '{}'".format(tar_path))

        if not isinstance(field_delim, str) or not field_delim:
            raise RuntimeError("'field_delim' should be a non-empty string, got '{}': {}"
                               .format(type(field_delim), field_delim))

        if not isinstance(write_buf_len, int) or write_buf_len < 0:
            print("WARING: 'write_buf_len' should be positive int, got '{}': {}, default to 0"
                  .format(type(write_buf_len), write_buf_len))
            write_buf_len = 0

        self.result_file_count = result_file_count
        self.tar_path = tar_path
        self.ret_file_name_prefix = (ret_file_name_prefix or '').strip()
        self.field_delim = field_delim
        self.write_buf_len = write_buf_len
        self.remove_empty_files = remove_empty_files
        self.header = header

    def _find_source_files(self):
        import glob
        source_files = []
        for pattern in self.source_file_patterns:
            files = glob.glob(pattern)
            print("found {} files from pattern '{}'".format(len(files), pattern))
            source_files.extend(files)

        print("totally found {} files from patterns {}".format(len(source_files), self.source_file_patterns))
        return sorted(source_files)

    def _read_thread(self, file, idx, queues: typing.List[queue.Queue], header_barrier: Barrier = None):
        try:
            csv_reader = CsvFileReader(file, self.field_delim)
            if self.header:
                if idx == 0:
                    headers = csv_reader.get_headers()
                    print("reader {}: read headers from '{}': {}".format(idx, file, headers))
                    for q in queues:
                        q.put(headers)
                if header_barrier is not None:
                    print("reader {}: waiting for reader 0 to send headers".format(idx))
                    header_barrier.wait()
                    print("reader {}: headers sent".format(idx))

            with csv_reader as reader:
                st = time.perf_counter()
                for i, row in enumerate(reader):
                    if self.header and i == 0:
                        continue
                    q_idx = (i-int(self.header)) % len(queues)
                    queues[q_idx].put(row)
                    if i % 10000 == 0:
                        print("reader {}: read {} lines from '{}', cost {}s"
                              .format(idx, i, file, time.perf_counter()-st))
                print("reader {}: read '{}' finished, totally {} lines, cost {}s"
                      .format(idx, file, i, time.perf_counter() - st))
        except Exception as e:
            import traceback
            print("reader {}: read '{}' error: {}\n{}".format(idx, file, e, traceback.format_exc()))
        finally:
            for q in queues:
                q.put(self.END_MARK)
            print("reader {}: sent end marks".format(idx))

    def _write_thread(self, file_name, idx, q: queue.Queue, num_sources):
        if self.write_buf_len > 0:
            buffer = []
            print("write {}: buffer length={}".format(idx, self.write_buf_len))
        else:
            print("write {}: no buffer".format(idx))
        st = time.perf_counter()
        write_counter = 0
        finish_sources = 0
        with CsvFileWriter(file_name, self.field_delim) as writer:
            while True:
                row = q.get()
                if isinstance(row, str) and row == self.END_MARK:
                    finish_sources += 1
                    print("writer {}: received {}/{} end marks".format(idx, finish_sources, num_sources))
                    if finish_sources >= num_sources:
                        if self.write_buf_len > 0 and buffer:
                            writer.writerows(buffer)
                            write_counter += len(buffer)
                            buffer.clear()
                        print("writer {}: all source finished, totally wrote {} lines to '{}', cost {}s"
                              .format(idx, write_counter, file_name, time.perf_counter()-st))
                        break
                    continue

                if self.write_buf_len > 0:
                    buffer.append(row)
                    if len(buffer) >= self.write_buf_len:
                        writer.writerows(buffer)
                        write_counter += len(buffer)
                        buffer.clear()
                else:
                    writer.writerow(row)
                    write_counter += 1

                if write_counter > 0 and write_counter % 10000 == 0:
                    print("writer {}: wrote {} lines to '{}', cost {}s".format(idx, write_counter, file_name,
                                                                               time.perf_counter()-st))

        if self.remove_empty_files and write_counter <= int(self.header):
            print("writer {}: no records for '{}'(except headers), will remove it".format(idx, file_name))
            os.remove(file_name)

    def shard(self):
        source_files = self._find_source_files()
        if not source_files:
            print("found no file to be sharded")
            return

        qs = [queue.Queue(self.write_buf_len) for _ in range(self.result_file_count)]
        read_tpool = ThreadPoolExecutor(len(source_files), 'csv_reader_tpool')
        write_tpool = ThreadPoolExecutor(self.result_file_count, 'csv_writer_tpool')
        st = time.perf_counter()
        if self.header and len(source_files) > 1:
            header_barrier = Barrier(len(source_files))
        else:
            header_barrier = None

        for i, source_file in enumerate(source_files):
            read_tpool.submit(self._read_thread, source_file, i, qs, header_barrier)

        path, base, ext = split_file_name(source_files[0])
        prefix = self.ret_file_name_prefix
        if not prefix and len(source_files) == 1:
            prefix = base + '-'
        elif prefix:
            prefix = prefix + '-'

        print("set result file name prefix='{}'".format(prefix))

        for j in range(self.result_file_count):
            tar_file_name = os.path.join(self.tar_path or path, prefix+"part-"+str(j)+ext)
            write_tpool.submit(self._write_thread, tar_file_name, j, qs[j], len(source_files))
        read_tpool.shutdown(wait=True)
        write_tpool.shutdown(wait=True)
        print("shard files finished, cost {}s".format(time.perf_counter()-st))


if __name__ == '__main__':
    sharder = CsvFileSharder('D:/work/docs/data/push/train_data_trans.csv', 3, ' ')
    sharder.shard()
