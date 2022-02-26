
import json
import typing
from collections import namedtuple, OrderedDict

import tensorflow as tf
import numpy as np

from job.pkgs.tf.feature_util import dtype_from_str, ModelInputConfig, InputDesc
from job.pkgs.utils import find_files


class TFRecordDesc(namedtuple('TFRecordDesc', ['name', 'ftype', 'shape', 'dtype_str',
                                               'default_value', 'allow_missing'])):
    FTYPE_MAP = {
        'fixlen': tf.io.FixedLenFeature,
        'varlen': tf.io.VarLenFeature,
        'seq': tf.io.FixedLenSequenceFeature
    }

    def __new__(cls, name, ftype, shape, dtype_str, default_value=None, allow_missing=None):
        return super(TFRecordDesc, cls).__new__(cls, name, ftype, shape, dtype_str, default_value, allow_missing)

    @property
    def dtype(self):
        return dtype_from_str(self.dtype_str)

    @classmethod
    def parse_from_json(cls, conf_json: typing.Dict):
        name = conf_json.get('name', '').strip()
        if not name:
            raise RuntimeError("'name' of tfrecord feature not set")

        ftype = conf_json.get('ftype', '').strip().lower()
        if not ftype or ftype not in cls.FTYPE_MAP:
            raise RuntimeError("unsupported tfrecord ftype '{}', supported are {}"
                               .format(ftype, cls.FTYPE_MAP.keys()))
        shape = conf_json.get('shape')
        if not shape:
            print("'shape' of tfrecord feature '{}' not set, assume it to be (1,)".format(name))
            shape = (1,)
        elif not isinstance(shape, (tuple, list)):
            shape = (int(shape),)
        else:
            shape = tuple(shape)

        if not all([isinstance(i, int) for i in shape]) or not all([i > 0 for i in shape]):
            raise RuntimeError("invalid shape {} of tfrecord feature '{}'".format(shape, name))

        dtype_str = conf_json.get('dtype', '').strip().lower()
        dtype = dtype_from_str(dtype_str)
        if dtype is None:
            raise RuntimeError("unknown dtype '{}' of tfrecord feature '{}'".format(dtype_str, name))

        default_value = conf_json.get('default_value')
        allow_missing = conf_json.get('allow_missing', False)

        return cls(name, ftype, shape, dtype_str, default_value, allow_missing)

    def to_tf_feature(self):
        feat_op = self.FTYPE_MAP.get(self.ftype)
        if feat_op == tf.io.FixedLenFeature:
            return feat_op(self.shape, self.dtype, self.default_value)
        if feat_op == tf.io.VarLenFeature:
            return feat_op(self.dtype)
        if feat_op == tf.io.FixedLenSequenceFeature:
            return feat_op(self.shape, self.dtype, self.allow_missing, self.default_value)

    def get_config(self):
        config = dict(self._asdict())
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TFRecordsConfig(object):
    def __init__(self, tfr_descs: typing.List[TFRecordDesc]):
        self._structure = {}
        for d in tfr_descs:
            if d.name in self._structure:
                raise RuntimeError("duplicated tfrecord feature name '{}'".format(d.name))
            self._structure[d.name] = d.to_tf_feature()
        self.tfr_descs = tfr_descs

    @property
    def structure(self):
        return self._structure

    @classmethod
    def parse(cls, json_file):
        with open(json_file, 'r') as f:
            config_json = json.load(f)
            if not isinstance(config_json, list):
                raise RuntimeError("tfrecord config file '{}' content should a json list, got '{}': {}"
                                   .format(json_file, type(config_json), config_json))

            tfr_descs = [TFRecordDesc.parse_from_json(d) for d in config_json]
        return cls(tfr_descs)

    def get_config(self):
        ser_tfr_descs = [d.get_config() for d in self.tfr_descs]
        return {'tfr_descs': ser_tfr_descs}

    @classmethod
    def from_config(cls, config):
        deser_tfr_descs = [TFRecordDesc.from_config(d) for d in config['tfr_descs']]
        return cls(deser_tfr_descs)


class TFRecordDataParser(object):
    def __init__(self, model_input_config_file, tfrecord_config_file, pack_path=None, export_path=None):
        self.model_input_config = ModelInputConfig.parse(model_input_config_file, pack_path, export_path)
        self.tfr_config = TFRecordsConfig.parse(tfrecord_config_file)

    def _parse_examples(self, records, label_names, sample_weight_name, sel_names=None, batch=True,
                        keep_label_name=None, fake_label=None, fake_label_rank=None):
        if batch:
            examples = tf.io.parse_example(records, self.tfr_config.structure, name='tfr_parse_example')
        else:
            examples = tf.io.parse_single_example(records, self.tfr_config.structure, name='tfr_parse_single_example')
        x = examples
        if sel_names:
            x = {n: examples[n] for n in sel_names}
        if label_names:
            if keep_label_name:
                y = {n: examples.pop(n) for n in label_names}
            else:
                y = tuple(examples.pop(n) for n in label_names)
                if len(y) == 1:
                    y = y[0]
        elif fake_label is not None:
            repeats = [tf.shape(list(x.values())[0])[0]] + [1] * fake_label_rank
            y = tf.tile([fake_label], repeats)

        if sample_weight_name:
            sw = examples.pop(sample_weight_name)

        if (fake_label is not None or label_names) and sample_weight_name:
            return x, y, sw
        elif fake_label is not None or label_names:
            return x, y
        return x

    def create_dataset_from_file(self, file_patterns, batch_size=None, compression_type=None,
                                 tfrecord_buffer_size=None, shuffle=False, shuffle_buffle_size=None,
                                 drop_remainder=False, input_groups=None, cache=False, repeat=False,
                                 deterministic=False, keep_label_name=None, fake_label=None):
        if input_groups:
            input_descs = self.model_input_config.get_inputs_by_group(input_groups)
        else:
            input_descs = self.model_input_config.all_inputs
        input_names = set([i.name for i in input_descs])
        dangled_input_names = input_names.difference(self.tfr_config.structure.keys())
        if dangled_input_names:
            raise RuntimeError("some input not included in tfrecord features: {}"
                               .format(dangled_input_names))

        label_descs = [d for d in input_descs if d.is_label]
        if not label_descs:
            print("WARING: found no label input")
            label_names = None
        else:
            if len(label_descs) > 1 and all([ld.label_index is not None for ld in label_descs]):
                label_descs = sorted(label_descs, key=lambda ld: ld.label_index)
                print("sorted labels: {}".format(label_descs))
            label_names = [ld.name for ld in label_descs]

        print("label_names={}".format(label_names))

        if keep_label_name is None:
            keep_label_name = label_names and len(label_names) > 1
            print("auto set keep_label_name={}".format(keep_label_name))

        sample_weight_desc = [d for d in input_descs if d.is_sample_weight]
        if sample_weight_desc and len(sample_weight_desc) > 1:
            raise RuntimeError("only 1 sample weight input is allowed, found {}: {}"
                               .format(len(sample_weight_desc), sample_weight_desc))
        elif sample_weight_desc:
            sample_weight_name = sample_weight_desc[0].name
        else:
            sample_weight_name = None

        print("sample_weight_name='{}'".format(sample_weight_name))

        data_files = find_files(file_patterns)
        if not data_files:
            raise RuntimeError("found no data files by patterns {}".format(file_patterns))

        print("found {} files by file patterns {}: {}".format(len(data_files), file_patterns, data_files[:10]))

        if batch_size is not None and batch_size <= 0:
            raise RuntimeError("batch_size should be > 0, got {}".format(batch_size))

        if compression_type is not None:
            compression_type = compression_type.strip().upper()
            if compression_type and compression_type not in ['ZLIB', 'GZIP']:
                raise NotImplementedError("'compression_type' should be ZLIB or GZIP, got {}"
                                          .format(compression_type))

        if shuffle and shuffle_buffle_size is not None and shuffle_buffle_size <= 0:
            raise RuntimeError("shuffle_buffle_size should be > 0, got {}".format(shuffle_buffle_size))

        if tfrecord_buffer_size is not None and tfrecord_buffer_size <= 0:
            raise RuntimeError("tfrecord_buffer_size should be > 0, got {}".format(tfrecord_buffer_size))

        if not label_names and fake_label is not None:
            fake_label_rank = np.array(fake_label).ndim
            print("will use fake label {}, rank={}".format(fake_label, fake_label_rank))
        else:
            fake_label_rank = None

        def __map_func(filename):
            return tf.data.TFRecordDataset(filename, compression_type=compression_type,
                                           buffer_size=tfrecord_buffer_size,
                                           num_parallel_reads=tf.data.experimental.AUTOTUNE)

        def __parse_func(records):
            return self._parse_examples(records, label_names, sample_weight_name, input_names,
                                        batch_size is not None, keep_label_name, fake_label, fake_label_rank)

        ds = tf.data.Dataset.from_tensor_slices(data_files)
        if repeat:
            ds = ds.repeat()
            print("[TFRecord]'{}': repeated dataset".format(file_patterns))
        if shuffle:
            if shuffle_buffle_size is not None:
                ds = ds.shuffle(shuffle_buffle_size)
            elif batch_size is not None:
                ds = ds.shuffle(batch_size*2)
            else:
                print("WARNING: neither shuffle_buffle_size or batch_size are set, ignore shuffle")
        ds = ds.interleave(__map_func, cycle_length=tf.data.experimental.AUTOTUNE,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                           deterministic=deterministic)

        if batch_size is not None:
            ds = ds.batch(batch_size, drop_remainder=drop_remainder)
            print("[TFRecord]'{}': batched dataset, batch_size={}, drop_remainder={}"
                  .format(file_patterns, batch_size, drop_remainder))
        ds = ds.map(__parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if cache:
            ds = ds.cache()
        return ds

        # ds = tf.data.Dataset.from_tensor_slices(data_files) \
        #     .interleave(__map_func,
        #                 cycle_length=tf.data.experimental.AUTOTUNE,
        #                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
        #                 deterministic=deterministic)
        #
        # if repeat:
        #     ds = ds.repeat()
        #     print("[CSV]'{}': repeated dataset".format(file_patterns))
        # if batch_size is not None and batch_size > 0:
        #     ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        #     print("[CSV]'{}': batched dataset, batch_size={}, drop_remainder={}"
        #           .format(file_patterns, batch_size, drop_remainder))
        # ds = ds.map(__parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # if shuffle:
        #     if shuffle_buffle_size is not None:
        #         ds = ds.shuffle(shuffle_buffle_size)
        #     elif batch_size is not None:
        #         ds = ds.shuffle(batch_size * 2)
        #     else:
        #         print("WARNING: neither shuffle_buffle_size or batch_size are set, ignore shuffle")
        # if cache:
        #     ds = ds.cache()
        # return ds


def extract_csv_input_header(input_descs: typing.List[InputDesc], file_patterns, field_delim,
                             with_header=True, headers=None):
    files = find_files(file_patterns)
    if not files:
        raise RuntimeError("no training data files were found with pattern '{}'".format(file_patterns))

    if with_header or headers:
        if with_header:
            import csv
            with open(files[0], 'r') as tmp:
                reader = csv.reader(tmp, delimiter=field_delim)
                all_cols = list(next(reader))
        else:
            if isinstance(headers, str):
                headers = headers.strip().split(',')
            all_cols = list(filter(lambda h: h, map(lambda h: h.strip(), headers)))
            if not all_cols:
                raise RuntimeError("invalid manual specified headers: '{}'".format(headers))
            print("manual specified headers: {}".format(all_cols))

        if len(all_cols) != len(set(all_cols)):
            raise RuntimeError("column names duplicated: {}".format(all_cols))

        col2idx = {c: i for i, c in enumerate(all_cols)}

        sel_col_ind_defs = {}
        for i_desc in input_descs:
            if i_desc.name not in col2idx:
                raise RuntimeError("input '{}' not contained in csv columns: {}, file_pattern='{}, field_delim='{}'"
                                   .format(i_desc.name, all_cols, file_patterns, field_delim))
            index = col2idx[i_desc.name]
            def_val = i_desc.infer_default_value()
            sel_col_ind_defs[index] = def_val

        sel_col_indices = []
        sel_col_defaults = []
        for i, v in sorted(sel_col_ind_defs.items(), key=lambda x: x[0]):
            sel_col_indices.append(i)
            sel_col_defaults.append(v)
    else:
        print("csv does not contain header and no header provided, assume headers be all model inputs")
        all_cols = [i.name for i in input_descs]
        sel_col_indices = list(range(len(input_descs)))
        sel_col_defaults = [i.infer_default_value() for i in input_descs]

    print("file_pattern='{}', field_delim='{}', parsed csv columns({}): {}, selected column indices({}): {},"
          " selected column defaults({}): {}".format(file_patterns, field_delim, len(all_cols), all_cols,
                                                     len(sel_col_indices), sel_col_indices, len(sel_col_defaults),
                                                     sel_col_defaults))

    return all_cols, sel_col_indices, sel_col_defaults


class CSVDataParser(object):
    def __init__(self, model_input_config_file, pack_path=None, export_path=None):
        self.model_input_config = ModelInputConfig.parse(model_input_config_file, pack_path, export_path)

    def create_dataset_from_file(self, file_patterns, field_delim, batch_size=None, shuffle=False,
                                 shuffle_buffle_size=None, drop_remainder=False, input_groups=None,
                                 with_headers=True, headers=None, cache=False, repeat=False,
                                 deterministic=False, keep_label_name=None, fake_label=None):
        if input_groups:
            input_descs = self.model_input_config.get_inputs_by_group(input_groups)
        else:
            input_descs = self.model_input_config.all_inputs

        label_descs = [d for d in input_descs if d.is_label]
        if not label_descs:
            print("WARING: found no label input")
            label_names = None
        else:
            if len(label_descs) > 1 and all([ld.label_index is not None for ld in label_descs]):
                label_descs = sorted(label_descs, key=lambda ld: ld.label_index)
                print("sorted labels: {}".format(label_descs))
            label_names = [ld.name for ld in label_descs]
        print("label_names={}".format(label_names))

        if keep_label_name is None:
            keep_label_name = label_names and len(label_names) > 1
            print("auto set keep_label_name={}".format(keep_label_name))

        sample_weight_desc = [d for d in input_descs if d.is_sample_weight]
        if sample_weight_desc and len(sample_weight_desc) > 1:
            raise RuntimeError("only 1 sample weight input is allowed, found {}: {}"
                               .format(len(sample_weight_desc), sample_weight_desc))
        elif sample_weight_desc:
            sample_weight_name = sample_weight_desc[0].name
        else:
            sample_weight_name = None

        print("sample_weight_name='{}'".format(sample_weight_name))

        data_files = find_files(file_patterns)
        if not data_files:
            raise RuntimeError("found no data files by patterns {}".format(file_patterns))

        print("found {} files by file patterns {}: {}".format(len(data_files), file_patterns, data_files[:10]))

        if not field_delim:
            raise RuntimeError("field_delim not set")

        if batch_size is not None and batch_size <= 0:
            raise RuntimeError("batch_size should be > 0, got {}".format(batch_size))

        if shuffle and shuffle_buffle_size is not None and shuffle_buffle_size <= 0:
            raise RuntimeError("shuffle_buffle_size should be > 0, got {}".format(shuffle_buffle_size))

        all_cols, sel_col_indices, sel_col_defauts = extract_csv_input_header(
            input_descs, file_patterns, field_delim, with_headers, headers)

        sel_cols = [all_cols[i] for i in sel_col_indices]
        label_indices = []
        sample_weight_index = None
        feature_names = []
        if label_names:
            for ln in label_names:
                for i, c in enumerate(sel_cols):
                    if c == ln:
                        label_indices.append(i)
                        break

        for i, c in enumerate(sel_cols):
            if label_names and c in label_names:
                continue
            if sample_weight_name and c == sample_weight_name:
                sample_weight_index = i
                continue
            feature_names.append(c)

        if not label_indices and fake_label is not None:
            fake_label_rank = np.array(fake_label).ndim
            print("will use fake label {}, rank={}".format(fake_label, fake_label_rank))

        print("label_indices={}, sample_weight_index={}, feature_names={}"
              .format(label_indices, sample_weight_index, feature_names))

        def __parse_func(line):
            fields = tf.io.decode_csv(line, sel_col_defauts, field_delim, select_cols=sel_col_indices,
                                      name='decode_csv_line')
            feature_fields = [fields[i] for i in range(len(sel_col_indices))
                              if i not in label_indices and i != sample_weight_index]

            x = OrderedDict(zip(feature_names, feature_fields))

            if label_indices:
                y = tuple(fields[i] for i in label_indices)
                if keep_label_name:
                    y = OrderedDict(zip(label_names, y))
                else:
                    if len(y) == 1:
                        y = y[0]
            elif fake_label is not None:
                repeats = [tf.shape(fields[0])[0]] + [1]*fake_label_rank
                y = tf.tile([fake_label], repeats)

            if sample_weight_index is not None:
                sw = fields[sample_weight_index]

            if (fake_label is not None or label_indices) and sample_weight_index is not None:
                return x, y, sw
            elif fake_label is not None or label_indices:
                return x, y
            return x

        def __map_func(filename):
            fnds = tf.data.TextLineDataset(filename, num_parallel_reads=tf.data.experimental.AUTOTUNE)
            if with_headers:
                fnds = fnds.skip(1)
            return fnds

        ds = tf.data.Dataset.from_tensor_slices(data_files)
        if repeat:
            ds = ds.repeat()
            print("[CSV]'{}': repeated dataset".format(file_patterns))
        if shuffle:
            if shuffle_buffle_size is not None:
                ds = ds.shuffle(shuffle_buffle_size)
            elif batch_size is not None:
                ds = ds.shuffle(batch_size * 2)
            else:
                print("WARNING: neither shuffle_buffle_size or batch_size are set, ignore shuffle")

        ds = ds.interleave(__map_func, cycle_length=tf.data.experimental.AUTOTUNE,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                           deterministic=deterministic)

        if batch_size is not None and batch_size > 0:
            ds = ds.batch(batch_size, drop_remainder=drop_remainder)
            print("[CSV]'{}': batched dataset, batch_size={}, drop_remainder={}"
                  .format(file_patterns, batch_size, drop_remainder))
        ds = ds.map(__parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if cache:
            ds = ds.cache()
        return ds

        # ds = tf.data.Dataset.from_tensor_slices(data_files) \
        #     .interleave(__map_func,
        #                 cycle_length=tf.data.experimental.AUTOTUNE,
        #                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
        #                 deterministic=deterministic)
        #
        # if repeat:
        #     ds = ds.repeat()
        #     print("[TFRecord]'{}': repeated dataset".format(file_patterns))
        # if batch_size is not None:
        #     ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        #     print("[TFRecord]'{}': batched dataset, batch_size={}, drop_remainder={}"
        #           .format(file_patterns, batch_size, drop_remainder))
        # ds = ds.map(__parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # if shuffle:
        #     if shuffle_buffle_size is not None:
        #         ds = ds.shuffle(shuffle_buffle_size)
        #     elif batch_size is not None:
        #         ds = ds.shuffle(batch_size * 2)
        #     else:
        #         print("WARNING: neither shuffle_buffle_size or batch_size are set, ignore shuffle")
        # if cache:
        #     ds = ds.cache()
        # return ds


def create_dataset(model_input_config_file, file_patterns, file_type=None, input_groups=None,
                   batch_size=None, shuffle=False, shuffle_buffle_size=None, drop_remainder=False,
                   cache=False, repeat=False, deterministic=False, keep_label_name=None, **kwargs):

    def __guess_file_type():
        files = find_files(file_patterns)
        if not files:
            raise RuntimeError("found no file by pattern '{}'".format(file_patterns))
        if any([f.lower().endswith('.csv') for f in files]):
            return 'CSV'
        return 'TFRECORD'

    if not file_type:
        file_type = __guess_file_type()
        if not file_type:
            raise RuntimeError("'file_type' not set")

    pack_path = kwargs.get('pack_path')
    data_path = kwargs.get('data_path')
    fake_label = kwargs.get('fake_label')
    file_type = file_type.strip().upper()
    if file_type == 'CSV':
        field_delim = kwargs.get('field_delim')
        with_headers = kwargs.get('with_headers')
        headers = kwargs.get('headers')
        if with_headers is None:
            if headers:
                with_headers = False
                print("user manually specified csv headers {}, set with_headers=False".format(headers))
            else:
                with_headers = True
                print("no csv headers specified, set with_headers=True")
        return CSVDataParser(model_input_config_file, pack_path, data_path)\
            .create_dataset_from_file(file_patterns, field_delim,
                                      batch_size, shuffle,
                                      shuffle_buffle_size, drop_remainder,
                                      input_groups, with_headers,
                                      headers, cache, repeat, deterministic,
                                      keep_label_name, fake_label=fake_label)
    elif file_type in ['TFR', 'TFRECORD']:
        tfrecord_config_file = kwargs.get('tfrecord_config_file')
        if not tfrecord_config_file:
            raise RuntimeError("'tfrecord_config_file' not set")
        compression_type = kwargs.get('compression_type')
        tfrecord_buffer_size = kwargs.get('tfrecord_buffer_size')
        return TFRecordDataParser(model_input_config_file, tfrecord_config_file, pack_path, data_path)\
            .create_dataset_from_file(file_patterns, batch_size, compression_type,
                                      tfrecord_buffer_size, shuffle, shuffle_buffle_size,
                                      drop_remainder, input_groups, cache, repeat,
                                      deterministic, keep_label_name, fake_label=fake_label)
