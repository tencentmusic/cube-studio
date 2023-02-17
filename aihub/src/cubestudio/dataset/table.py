import copy
import os
import tempfile
import warnings
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import pysnooper
import shutil
import contextlib
import pyarrow.compute as pc
from collections import Counter
from collections.abc import Mapping
import json,sys,io
from multiprocess import Pool, RLock
import numpy as np
import pandas as pd
import pyarrow as pa

from tqdm.auto import tqdm
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
from . import config
from .utils import logging
from .utils.logging import get_logger


# if TYPE_CHECKING:
from .features.features import Features, FeatureType


logger = get_logger(__name__)


def inject_arrow_table_documentation(arrow_table_method):
    def wrapper(fn):
        fn.__doc__ = arrow_table_method.__doc__ + (fn.__doc__ if fn.__doc__ is not None else "")
        fn.__doc__ = fn.__doc__.replace("pyarrow.Table", "Table")
        if hasattr(arrow_table_method, "__annotations__"):
            fn.__annotations__ = arrow_table_method.__annotations__
        return fn

    return wrapper


def _in_memory_arrow_table_from_file(filename: str) -> pa.Table:
    in_memory_stream = pa.input_stream(filename)
    opened_stream = pa.ipc.open_stream(in_memory_stream)
    pa_table = opened_stream.read_all()
    return pa_table


def _in_memory_arrow_table_from_buffer(buffer: pa.Buffer) -> pa.Table:
    stream = pa.BufferReader(buffer)
    opened_stream = pa.ipc.open_stream(stream)
    table = opened_stream.read_all()
    return table


def _memory_mapped_arrow_table_from_file(filename: str) -> pa.Table:
    memory_mapped_stream = pa.memory_map(filename)
    opened_stream = pa.ipc.open_stream(memory_mapped_stream)
    pa_table = opened_stream.read_all()
    return pa_table


def _write_table_to_file(table: pa.Table, filename: str) -> int:
    with open(filename, "wb") as sink:
        writer = pa.RecordBatchStreamWriter(sink=sink, schema=table.schema)
        batches: List[pa.RecordBatch] = table.to_batches()
        for batch in batches:
            writer.write_batch(batch)
        writer.close()
        return sum(batch.nbytes for batch in batches)

# 对象的深度复制
def _deepcopy(x, memo: dict):
    """deepcopy a regular class instance"""
    cls = x.__class__
    result = cls.__new__(cls)
    memo[id(x)] = result
    for k, v in x.__dict__.items():
        setattr(result, k, copy.deepcopy(v, memo))
    return result

def _interpolation_search(arr: List[int], x: int) -> int:
    """
    Return the position i of a sorted array so that arr[i] <= x < arr[i+1]

    Args:
        arr (`List[int]`): non-empty sorted list of integers
        x (`int`): query

    Returns:
        `int`: the position i so that arr[i] <= x < arr[i+1]

    Raises:
        `IndexError`: if the array is empty or if the query is outside the array values
    """
    i, j = 0, len(arr) - 1
    while i < j and arr[i] <= x < arr[j]:
        k = i + ((j - i) * (x - arr[i]) // (arr[j] - arr[i]))
        if arr[k] <= x < arr[k + 1]:
            return k
        elif arr[k] < x:
            i, j = k + 1, j
        else:
            i, j = i, k
    raise IndexError(f"Invalid query '{x}' for size {arr[-1] if len(arr) else 'none'}.")


class IndexedTableMixin:
    def __init__(self, table: pa.Table):
        self._schema: pa.Schema = table.schema
        self._batches: List[pa.RecordBatch] = [
            recordbatch for recordbatch in table.to_batches() if len(recordbatch) > 0
        ]
        self._offsets: np.ndarray = np.cumsum([0] + [len(b) for b in self._batches], dtype=np.int64)

    def fast_gather(self, indices: Union[List[int], np.ndarray]) -> pa.Table:
        """
        Create a pa.Table by gathering the records at the records at the specified indices. Should be faster
        than pa.concat_tables(table.fast_slice(int(i) % table.num_rows, 1) for i in indices) since NumPy can compute
        the binary searches in parallel, highly optimized C
        """
        if not len(indices):
            raise ValueError("Indices must be non-empty")
        batch_indices = np.searchsorted(self._offsets, indices, side="right") - 1
        return pa.Table.from_batches(
            [
                self._batches[batch_idx].slice(i - self._offsets[batch_idx], 1)
                for batch_idx, i in zip(batch_indices, indices)
            ],
            schema=self._schema,
        )

    def fast_slice(self, offset=0, length=None) -> pa.Table:
        """
        Slice the Table using interpolation search.
        The behavior is the same as `pyarrow.Table.slice` but it's significantly faster.

        Interpolation search is used to find the start and end indexes of the batches we want to keep.
        The batches to keep are then concatenated to form the sliced Table.
        """
        if offset < 0:
            raise IndexError("Offset must be non-negative")
        elif offset >= self._offsets[-1] or (length is not None and length <= 0):
            return pa.Table.from_batches([], schema=self._schema)
        i = _interpolation_search(self._offsets, offset)
        if length is None or length + offset >= self._offsets[-1]:
            batches = self._batches[i:]
            batches[0] = batches[0].slice(offset - self._offsets[i])
        else:
            j = _interpolation_search(self._offsets, offset + length - 1)
            batches = self._batches[i : j + 1]
            batches[-1] = batches[-1].slice(0, offset + length - self._offsets[j])
            batches[0] = batches[0].slice(offset - self._offsets[i])
        return pa.Table.from_batches(batches, schema=self._schema)


class _RecordBatchReader:
    def __init__(self, table: "Table", max_chunksize: Optional[int] = None):
        self.table = table
        self.max_chunksize = max_chunksize

    def __iter__(self):
        for batch in self.table._batches:
            if self.max_chunksize is None or len(batch) <= self.max_chunksize:
                yield batch
            else:
                for offset in range(0, len(batch), self.max_chunksize):
                    yield batch.slice(offset, self.max_chunksize)


class Table(IndexedTableMixin):
    """
    Wraps a pyarrow Table by using composition.
    This is the base class for `InMemoryTable`, `MemoryMappedTable` and `ConcatenationTable`.

    It implements all the basic attributes/methods of the pyarrow Table class except
    the Table transforms: `slice, filter, flatten, combine_chunks, cast, add_column,
    append_column, remove_column, set_column, rename_columns` and `drop`.

    The implementation of these methods differs for the subclasses.
    """

    def __init__(self, table: pa.Table,features=None):
        super().__init__(table)
        self.table = table
        if features:
            pa_metadata = {"cube-studio": json.dumps({"info":{"features":features.to_dict()}})}
            self.table.replace_schema_metadata(pa_metadata)

    def __deepcopy__(self, memo: dict):
        # arrow tables are immutable, so there's no need to copy self.table
        # moreover calling deepcopy on a pyarrow table seems to make pa.total_allocated_bytes() decrease for some reason
        # by adding it to the memo, self.table won't be copied
        memo[id(self.table)] = self.table
        # same for the recordbatches used by the index
        memo[id(self._batches)] = list(self._batches)
        return _deepcopy(self, memo)

    def __getstate__(self):
        # We can't pickle objects that are bigger than 4GiB, or it causes OverflowError
        # So we write the table on disk instead
        if self.table.nbytes >= config.MAX_TABLE_NBYTES_FOR_PICKLING:
            table = self.table
            with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".arrow") as tmp_file:
                filename = tmp_file.name
                logger.debug(
                    f"Attempting to pickle a table bigger than 4GiB. Writing it on the disk instead at {filename}"
                )
                _write_table_to_file(table=table, filename=filename)
                return {"path": filename}
        else:
            return {"table": self.table}

    def __setstate__(self, state):
        if "path" in state:
            filename = state["path"]
            logger.debug(f"Unpickling a big table from the disk at {filename}")
            table = _in_memory_arrow_table_from_file(filename)
            logger.debug(f"Removing temporary table file at {filename}")
            os.remove(filename)
        else:
            table = state["table"]
        Table.__init__(self, table)

    def validate(self, *args, **kwargs):
        """
        Perform validation checks.  An exception is raised if validation fails.

        By default only cheap validation checks are run.  Pass `full=True`
        for thorough validation checks (potentially `O(n)`).

        Args:
            full (`bool`, defaults to `False`):
                If `True`, run expensive checks, otherwise cheap checks only.

        Raises:
            `pa.lib.ArrowInvalid`: if validation fails
        """
        return self.table.validate(*args, **kwargs)


    def equals(self, *args, **kwargs):
        """
        Check if contents of two tables are equal.

        Args:
            other ([`~datasets.table.Table`]):
                Table to compare against.
            check_metadata `bool`, defaults to `False`):
                Whether schema metadata equality should be checked as well.

        Returns:
            `bool`
        """
        args = tuple(arg.table if isinstance(arg, Table) else arg for arg in args)
        kwargs = {k: v.table if isinstance(v, Table) else v for k, v in kwargs}
        return self.table.equals(*args, **kwargs)

    def to_batches(self, *args, **kwargs):
        """
        Convert Table to list of (contiguous) `RecordBatch` objects.

        Args:
            max_chunksize (`int`, defaults to `None`):
                Maximum size for `RecordBatch` chunks. Individual chunks may be
                smaller depending on the chunk layout of individual columns.

        Returns:
            `List[pyarrow.RecordBatch]`
        """
        return self.table.to_batches(*args, **kwargs)

    def to_pydict(self, *args, **kwargs):
        """
        Convert the Table to a `dict` or `OrderedDict`.

        Returns:
            `dict`
        """
        return self.table.to_pydict(*args, **kwargs)

    def to_pylist(self, *args, **kwargs):
        """
        Convert the Table to a list

        Returns:
            `list`
        """
        try:
            return self.table.to_pylist(*args, **kwargs)
        except AttributeError:  # pyarrow <7 does not have to_pylist, so we use to_pydict
            pydict = self.table.to_pydict(*args, **kwargs)
            return [{k: pydict[k][i] for k in pydict} for i in range(len(self.table))]

    def to_pandas(self, *args, **kwargs):
        """
        Convert to a pandas-compatible NumPy array or DataFrame, as appropriate.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                Arrow MemoryPool to use for allocations. Uses the default memory
                pool is not passed.
            strings_to_categorical (`bool`, defaults to `False`):
                Encode string (UTF8) and binary types to `pandas.Categorical`.
            categories (`list`, defaults to `empty`):
                List of fields that should be returned as `pandas.Categorical`. Only
                applies to table-like data structures.
            zero_copy_only (`bool`, defaults to `False`):
                Raise an `ArrowException` if this function call would require copying
                the underlying data.
            integer_object_nulls (`bool`, defaults to `False`):
                Cast integers with nulls to objects.
            date_as_object (`bool`, defaults to `True`):
                Cast dates to objects. If `False`, convert to `datetime64[ns]` dtype.
            timestamp_as_object (`bool`, defaults to `False`):
                Cast non-nanosecond timestamps (`np.datetime64`) to objects. This is
                useful if you have timestamps that don't fit in the normal date
                range of nanosecond timestamps (1678 CE-2262 CE).
                If `False`, all timestamps are converted to `datetime64[ns]` dtype.
            use_threads (`bool`, defaults to `True`):
                Whether to parallelize the conversion using multiple threads.
            deduplicate_objects (`bool`, defaults to `False`):
                Do not create multiple copies Python objects when created, to save
                on memory use. Conversion will be slower.
            ignore_metadata (`bool`, defaults to `False`):
                If `True`, do not use the 'pandas' metadata to reconstruct the
                DataFrame index, if present.
            safe (`bool`, defaults to `True`):
                For certain data types, a cast is needed in order to store the
                data in a pandas DataFrame or Series (e.g. timestamps are always
                stored as nanoseconds in pandas). This option controls whether it
                is a safe cast or not.
            split_blocks (`bool`, defaults to `False`):
                If `True`, generate one internal "block" for each column when
                creating a pandas.DataFrame from a `RecordBatch` or `Table`. While this
                can temporarily reduce memory note that various pandas operations
                can trigger "consolidation" which may balloon memory use.
            self_destruct (`bool`, defaults to `False`):
                EXPERIMENTAL: If `True`, attempt to deallocate the originating Arrow
                memory while converting the Arrow object to pandas. If you use the
                object after calling `to_pandas` with this option it will crash your
                program.
            types_mapper (`function`, defaults to `None`):
                A function mapping a pyarrow DataType to a pandas `ExtensionDtype`.
                This can be used to override the default pandas type for conversion
                of built-in pyarrow types or in absence of `pandas_metadata` in the
                Table schema. The function receives a pyarrow DataType and is
                expected to return a pandas `ExtensionDtype` or `None` if the
                default conversion should be used for that type. If you have
                a dictionary mapping, you can pass `dict.get` as function.

        Returns:
            `pandas.Series` or `pandas.DataFrame`: `pandas.Series` or `pandas.DataFrame` depending on type of object
        """
        return self.table.to_pandas(*args, **kwargs)

    def to_string(self, *args, **kwargs):
        return self.table.to_string(*args, **kwargs)

    def to_reader(self, max_chunksize: Optional[int] = None):
        """
        Convert the Table to a RecordBatchReader.

        Note that this method is zero-copy, it merely exposes the same data under a different API.

        Args:
            max_chunksize (`int`, defaults to `None`)
                Maximum size for RecordBatch chunks. Individual chunks may be smaller depending
                on the chunk layout of individual columns.

        Returns:
            `pyarrow.RecordBatchReader` if pyarrow>=8.0.0, otherwise a `pyarrow.RecordBatch` iterable
        """
        if config.PYARROW_VERSION.major < 8:
            return _RecordBatchReader(self, max_chunksize=max_chunksize)
        return self.table.to_reader(max_chunksize=max_chunksize)

    def field(self, *args, **kwargs):
        """
        Select a schema field by its column name or numeric index.

        Args:
            i (`Union[int, str]`):
                The index or name of the field to retrieve.

        Returns:
            `pyarrow.Field`
        """
        return self.table.field(*args, **kwargs)

    def column(self, *args, **kwargs):
        """
        Select a column by its column name, or numeric index.

        Args:
            i (`Union[int, str]`):
                The index or name of the column to retrieve.

        Returns:
            `pyarrow.ChunkedArray`
        """
        return self.table.column(*args, **kwargs)

    def itercolumns(self, *args, **kwargs):
        """
        Iterator over all columns in their numerical order.

        Yields:
            `pyarrow.ChunkedArray`
        """
        return self.table.itercolumns(*args, **kwargs)

    @property
    def schema(self):
        """
        Schema of the table and its columns.

        Returns:
            `pyarrow.Schema`
        """
        return self.table.schema

    @property
    def info(self):
        info = f'''
num_columns: {self.num_columns}
num_rows: {self.num_rows}
column_names:{self.column_names}
shape: {self.shape}
        '''
        return info


    @property
    def columns(self):
        """
        List of all columns in numerical order.

        Returns:
            `List[pa.ChunkedArray]`
        """
        return self.table.columns

    @property
    def num_columns(self):
        """
        Number of columns in this table.

        Returns:
            int
        """
        return self.table.num_columns

    @property
    def num_rows(self):
        """
        Number of rows in this table.

        Due to the definition of a table, all columns have the same number of
        rows.

        Returns:
            int
        """
        return self.table.num_rows

    @property
    def shape(self):
        """
        Dimensions of the table: (#rows, #columns).

        Returns:
            `(int, int)`: Number of rows and number of columns.
        """
        return self.table.shape

    @property
    def nbytes(self):
        """
        Total number of bytes consumed by the elements of the table.
        """
        return self.table.nbytes

    @property
    def column_names(self):
        """
        Names of the table's columns.
        """
        return self.table.column_names

    def __eq__(self, other):
        return self.equals(other)

    def __getitem__(self, i):
        return self.table[i]

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        return self.table.__repr__().replace("pyarrow.Table", self.__class__.__name__)

    def __str__(self):
        return self.table.__str__().replace("pyarrow.Table", self.__class__.__name__)



    def slice(self, *args, **kwargs):
        """
        Compute zero-copy slice of this Table.

        Args:
            offset (`int`, defaults to `0`):
                Offset from start of table to slice.
            length (`int`, defaults to `None`):
                Length of slice (default is until end of table starting from
                offset).

        Returns:
            `datasets.table.Table`
        """
        raise NotImplementedError()

    def filter(self, *args, **kwargs):
        """
        Select records from a Table. See `pyarrow.compute.filter` for full usage.
        """
        raise NotImplementedError()

    def flatten(self, *args, **kwargs):
        """
        Flatten this Table.  Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        raise NotImplementedError()

    def combine_chunks(self, *args, **kwargs):
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the `ChunkedArray` of each column are
        concatenated into zero or one chunk.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        raise NotImplementedError()

    def cast(self, *args, **kwargs):
        """
        Cast table values to another schema.

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        raise NotImplementedError()

    def replace_schema_metadata(self, *args, **kwargs):
        """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be None,
        which deletes any existing metadata

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
        raise NotImplementedError()

    def add_column(self, *args, **kwargs):
        """
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column added.
        """
        raise NotImplementedError()

    def append_column(self, *args, **kwargs):
        """
        Append column at end of columns.

        Args:
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:  New table with the passed column added.
        """
        raise NotImplementedError()

    def remove_column(self, *args, **kwargs):
        """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`: New table without the column.
        """
        raise NotImplementedError()

    def set_column(self, *args, **kwargs):
        """
        Replace column in Table at position.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column set.
        """
        raise NotImplementedError()

    def rename_columns(self, *args, **kwargs):
        """
        Create new table with columns renamed to provided names.
        """
        raise NotImplementedError()

    def drop(self, *args, **kwargs):
        """
        Drop one or more columns and return a new table.

        Args:
            columns (`List[str]`):
                List of field names referencing existing columns.

        Raises:
            `KeyError` : if any of the passed columns name are not existing.

        Returns:
            `datasets.table.Table`: New table without the columns.
        """
        raise NotImplementedError()

    def select(self, *args, **kwargs):
        """
        Select columns of the table.

        Returns a new table with the specified columns, and metadata preserved.

        Args:
            columns (:obj:`Union[List[str], List[int]]`):
                The column names or integer indices to select.

        Returns:
            `datasets.table.Table`: table with only a subset of the columns
        """
        raise NotImplementedError()

    def sort_by(self, *args, **kwargs):
        raise NotImplementedError()
    # 合并另一个表
    def concat_table(self, table):
        raise NotImplementedError()

class TableBlock(Table):
    """
    `TableBlock` is the allowed class inside a `ConcanetationTable`.
    Only `MemoryMappedTable` and `InMemoryTable` are `TableBlock`.
    This is because we don't want a `ConcanetationTable` made out of other `ConcanetationTables`.
    """

    pass


class InMemoryTable(TableBlock):
    """
    The table is said in-memory when it is loaded into the user's RAM.

    Pickling it does copy all the data using memory.
    Its implementation is simple and uses the underlying pyarrow Table methods directly.

    This is different from the `MemoryMapped` table, for which pickling doesn't copy all the
    data in memory. For a `MemoryMapped`, unpickling instead reloads the table from the disk.

    `InMemoryTable` must be used when data fit in memory, while `MemoryMapped` are reserved for
    data bigger than memory or when you want the memory footprint of your application to
    stay low.
    """

    @classmethod
    def from_file(cls, filename: str):
        table = _in_memory_arrow_table_from_file(filename)
        return cls(table)

    @classmethod
    def from_buffer(cls, buffer: pa.Buffer):
        table = _in_memory_arrow_table_from_buffer(buffer)
        return cls(table)

    @classmethod
    def from_pandas(cls, *args, **kwargs):
        """
        Convert pandas.DataFrame to an Arrow Table.

        The column types in the resulting Arrow Table are inferred from the
        dtypes of the pandas.Series in the DataFrame. In the case of non-object
        Series, the NumPy dtype is translated to its Arrow equivalent. In the
        case of `object`, we need to guess the datatype by looking at the
        Python objects in this Series.

        Be aware that Series of the `object` dtype don't carry enough
        information to always lead to a meaningful Arrow type. In the case that
        we cannot infer a type, e.g. because the DataFrame is of length 0 or
        the Series only contains `None/nan` objects, the type is set to
        null. This behavior can be avoided by constructing an explicit schema
        and passing it to this function.

        Args:
            df (`pandas.DataFrame`):
            schema (`pyarrow.Schema`, *optional*):
                The expected schema of the Arrow Table. This can be used to
                indicate the type of columns if we cannot infer it automatically.
                If passed, the output will have exactly this schema. Columns
                specified in the schema that are not found in the DataFrame columns
                or its index will raise an error. Additional columns or index
                levels in the DataFrame which are not specified in the schema will
                be ignored.
            preserve_index (`bool`, *optional*):
                Whether to store the index as an additional column in the resulting
                `Table`. The default of None will store the index as a column,
                except for RangeIndex which is stored as metadata only. Use
                `preserve_index=True` to force it to be stored as a column.
            nthreads (`int`, defaults to `None` (may use up to system CPU count threads))
                If greater than 1, convert columns to Arrow in parallel using
                indicated number of threads.
            columns (`List[str]`, *optional*):
               List of column to be converted. If `None`, use all columns.
            safe (`bool`, defaults to `True`):
               Check for overflows or other unsafe conversions,

        Returns:
            `datasets.table.Table`:

        Examples:
        ```python
        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> df = pd.DataFrame({
            ...     'int': [1, 2],
            ...     'str': ['a', 'b']
            ... })
        >>> pa.Table.from_pandas(df)
        <pyarrow.lib.Table object at 0x7f05d1fb1b40>
        ```
        """
        return cls(pa.Table.from_pandas(*args, **kwargs))

    @classmethod
    def from_arrays(cls, *args, **kwargs):
        """
        Construct a Table from Arrow arrays.

        Args:
            arrays (`List[Union[pyarrow.Array, pyarrow.ChunkedArray]]`):
                Equal-length arrays that should form the table.
            names (`List[str]`, *optional*):
                Names for the table columns. If not passed, schema must be passed.
            schema (`Schema`, defaults to `None`):
                Schema for the created table. If not passed, names must be passed.
            metadata (`Union[dict, Mapping]`, defaults to `None`):
                Optional metadata for the schema (if inferred).

        Returns:
            `datasets.table.Table`
        """
        return cls(pa.Table.from_arrays(*args, **kwargs))

    @classmethod
    def from_pydict(cls, *args, **kwargs):
        """
        Construct a Table from Arrow arrays or columns.

        Args:
            mapping (`Union[dict, Mapping]`):
                A mapping of strings to Arrays or Python lists.
            schema (`Schema`, defaults to `None`):
                If not passed, will be inferred from the Mapping values
            metadata (`Union[dict, Mapping]`, defaults to `None`):
                Optional metadata for the schema (if inferred).

        Returns:
            `datasets.table.Table`
        """
        return cls(pa.Table.from_pydict(*args, **kwargs))

    @classmethod
    def from_pylist(cls, mapping, *args, **kwargs):
        """
        Construct a Table from list of rows / dictionaries.

        Args:
            mapping (`List[dict]`):
                A mapping of strings to row values.
            schema (`Schema`, defaults to `None`):
                If not passed, will be inferred from the Mapping values
            metadata (`Union[dict, Mapping]`, defaults to `None`):
                Optional metadata for the schema (if inferred).

        Returns:
            `datasets.table.Table`
        """
        try:
            return cls(pa.Table.from_pylist(mapping, *args, **kwargs))
        except AttributeError:  # pyarrow <7 does not have from_pylist, so we convert and use from_pydict
            mapping = {k: [r.get(k) for r in mapping] for k in mapping[0]} if mapping else {}
            return cls(pa.Table.from_pydict(mapping, *args, **kwargs))

    @classmethod
    def from_batches(cls, *args, **kwargs):
        """
        Construct a Table from a sequence or iterator of Arrow `RecordBatches`.

        Args:
            batches (`Union[Sequence[pyarrow.RecordBatch], Iterator[pyarrow.RecordBatch]]`):
                Sequence of `RecordBatch` to be converted, all schemas must be equal.
            schema (`Schema`, defaults to `None`):
                If not passed, will be inferred from the first `RecordBatch`.

        Returns:
            `datasets.table.Table`:
        """
        return cls(pa.Table.from_batches(*args, **kwargs))


    @property
    def cache_files(self) -> List[dict]:
        """The cache files containing the Apache Arrow table backing the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.cache_files
        [{'filename': '/root/.cache/huggingface/datasets/rotten_tomatoes_movie_review/default/1.0.0/40d411e45a6ce3484deed7cc15b82a53dad9a72aafd9f86f8f227134bec5ca46/rotten_tomatoes_movie_review-validation.arrow'}]
        ```
        """
        cache_files = list_table_cache_files(self)
        return [{"filename": cache_filename} for cache_filename in cache_files]

    def slice(self, offset=0, length=None):
        """
        Compute zero-copy slice of this Table.

        Args:
            offset (`int`, defaults to `0`):
                Offset from start of table to slice.
            length (`int`, defaults to `None`):
                Length of slice (default is until end of table starting from
                offset).

        Returns:
            `datasets.table.Table`
        """
        # Use fast slicing here
        return InMemoryTable(self.fast_slice(offset=offset, length=length))

    def filter(self, *args, **kwargs):
        """
        Select records from a Table. See `pyarrow.compute.filter` for full usage.
        """
        return InMemoryTable(self.table.filter(*args, **kwargs))

    def flatten(self, *args, **kwargs):
        """
        Flatten this Table.  Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        return InMemoryTable(self.table.flatten())

    def combine_chunks(self, *args, **kwargs):
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the `ChunkedArray` of each column are
        concatenated into zero or one chunk.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        return InMemoryTable(self.table.combine_chunks(*args, **kwargs))

    def cast(self, *args, **kwargs):
        """
        Cast table values to another schema.

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        return InMemoryTable(table_cast(self.table, *args, **kwargs))

    @property
    def features(self):
        from .features.features import Features
        features = Features.from_arrow_schema(self.table.schema)
        return features

    # def renew_features(self):
    #     from .features.features import Features
    #     features = Features.from_arrow_schema(self.table.schema)
    #     return features

    # @pysnooper.snoop()
    def cast_column(self, column: str, feature):
        """
        Cast table values to another schema.

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        features = self.features
        features[column] = feature
        table = InMemoryTable(table_cast(self.table, features.arrow_schema))
        return table

    def replace_schema_metadata(self, *args, **kwargs):
        """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be `None`,
        which deletes any existing metadata).

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
        return InMemoryTable(self.table.replace_schema_metadata(*args, **kwargs))

    def add_column(self, *args, **kwargs):
        """
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column added.
        """
        return InMemoryTable(self.table.add_column(*args, **kwargs))

    def append_column(self, *args, **kwargs):
        """
        Append column at end of columns.

        Args:
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column added.
        """
        return InMemoryTable(self.table.append_column(*args, **kwargs))

    def remove_column(self, *args, **kwargs):
        """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`:
                New table without the column.
        """
        return InMemoryTable(self.table.remove_column(*args, **kwargs))

    def set_column(self, *args, **kwargs):
        """
        Replace column in Table at position.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column set.
        """
        return InMemoryTable(self.table.set_column(*args, **kwargs))

    def rename_columns(self, column_mapping,*args, **kwargs):
        """
        Create new table with columns renamed to provided names.
        """
        if type(column_mapping)==dict:
            extra_columns = [column_mapping[x] if x in column_mapping else x for x in self.table.column_names]
            args = [extra_columns]
        if type(column_mapping)==list:
            args=[column_mapping]

        return InMemoryTable(self.table.rename_columns(*args, **kwargs))

    def drop(self, *args, **kwargs):
        """
        Drop one or more columns and return a new table.

        Args:
            columns (`List[str]`):
                List of field names referencing existing columns.

        Raises:
            `KeyError` : if any of the passed columns name are not existing.

        Returns:
            `datasets.table.Table`:
                New table without the columns.
        """
        return InMemoryTable(self.table.drop(*args, **kwargs))

    def select(self, *args, **kwargs):
        """
        Select columns of the table.

        Returns a new table with the specified columns, and metadata preserved.

        Args:
            columns (:obj:`Union[List[str], List[int]]`):
                The column names or integer indices to select.

        Returns:
            :class:`datasets.table.Table`: New table with the specified columns, and metadata preserved.
        """
        return InMemoryTable(self.table.select(*args, **kwargs))

    def sort_by(self, *args, **kwargs):
        return InMemoryTable(self.table.sort_by(*args, **kwargs))

    # 合并另一个表
    def concat_table(self, table):
        pa_table = pa.concat_tables([self.table,table.table])
        return InMemoryTable(pa_table)

    def __getitem__(self, key):
        if type(key)==str:
            return self.table[key]  # 索引列
        if type(key)==int:   # 索引行
            return self.table.slice(key,1)

        if type(key)==tuple and len(key)>0:
            if type(key[0])==str:
                return self.select(list(key))  # 索引多列
            if type(key[0])==int and len(key)==2:  # 索引多行
                return self.table.slice(*key)

    # @pysnooper.snoop()
    # 索引数据的时候，进行decode_example的调用。
    def _getitem(self, key: Union[int, slice, str], **kwargs) -> Union[Dict, List]:
        from .formatting import format_table, get_formatter, query_table
        formatter = get_formatter(None, features=self.features)
        pa_subtable = query_table(self, key, None)
        formatted_output = format_table(
            pa_subtable, key, formatter=formatter, format_columns=None, output_all_columns=False
        )
        return formatted_output

    # @pysnooper.snoop()
    def __iter__(self):
        """Iterate through the examples.

        If a formatting is set with :meth:`Dataset.set_format` rows will be returned with the
        selected format.
        """
        from .formatting import format_table, get_formatter
        format_kwargs = {}
        formatter = get_formatter(self._format_type, features=self.features, **format_kwargs)
        batch_size = config.ARROW_READER_BATCH_SIZE_IN_DATASET_ITER
        for pa_subtable in table_iter(self, batch_size=batch_size):
            # print(type(pa_subtable))
            for i in range(pa_subtable.num_rows):
                pa_subtable_ex = pa_subtable.slice(i, 1)
                self._format_columns=None
                self._output_all_columns = False
                formatted_output = format_table(
                    pa_subtable_ex,
                    0,
                    formatter=formatter,
                    format_columns=self._format_columns,
                    output_all_columns=self._output_all_columns,
                )
                yield formatted_output

    def __repr__(self):
        return f"{self.__class__.__name__} ({{\n    features: {self.features},\n    num_rows: {self.num_rows}\n}})"

    def __str__(self):
        return f"{self.__class__.__name__} ({{\n    features: {self.features},\n    num_rows: {self.num_rows}\n}})"


    # @pysnooper.snoop()
    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, List[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> "Table":
        """
        Apply a function to all the examples in the table (individually or in batches) and update the table.
        If your function returns a column that already exists, then it overwrites it.

        You can specify whether the function should be batched or not with the `batched` parameter:

        - If batched is `False`, then the function takes 1 example in and should return 1 example.
          An example is a dictionary, e.g. `{"text": "Hello there !"}`.
        - If batched is `True` and `batch_size` is 1, then the function takes a batch of 1 example as input and can return a batch with 1 or more examples.
          A batch is a dictionary, e.g. a batch of 1 example is `{"text": ["Hello there !"]}`.
        - If batched is `True` and `batch_size` is `n > 1`, then the function takes a batch of `n` examples as input and can return a batch with `n` examples, or with an arbitrary number of examples.
          Note that the last batch may have less than `n` examples.
          A batch is a dictionary, e.g. a batch of `n` examples is `{"text": ["Hello there !"] * n}`.

        Args:
            function (`Callable`): Function with one of the following signatures:

                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False` and `with_rank=False`
                - `function(example: Dict[str, Any], *extra_args) -> Dict[str, Any]` if `batched=False` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)
                - `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False` and `with_rank=False`
                - `function(batch: Dict[str, List], *extra_args) -> Dict[str, List]` if `batched=True` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)

                For advanced usage, the function can also return a `pyarrow.Table`.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the dataset unchanged.
                If no function is provided, default to identity function: `lambda x: x`.
            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the
                signature of `function` should be `def function(example, idx[, rank]): ...`.
            with_rank (`bool`, defaults to `False`):
                Provide process rank to `function`. Note that in this case the
                signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`Optional[Union[str, List[str]]]`, defaults to `None`):
                The columns to be passed into `function`
                as positional arguments. If `None`, a `dict` mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`.
                If `batch_size <= 0` or `batch_size == None`, provide the full dataset as a single batch to `function`.
            drop_last_batch (`bool`, defaults to `False`):
                Whether a last batch smaller than the batch_size should be
                dropped instead of being processed by the function.
            remove_columns (`Optional[Union[str, List[str]]]`, defaults to `None`):
                Remove a selection of columns while doing the mapping.
                Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
                columns with names in `remove_columns`, these columns will be kept.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            load_from_cache_file (`bool`, defaults to `True` if caching is enabled):
                If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            cache_file_name (`str`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            features (`Optional[datasets.Features]`, defaults to `None`):
                Use a specific Features to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`):
                Disallow null values in the table.
            fn_kwargs (`Dict`, *optional*, defaults to `None`):
                Keyword arguments to be passed to `function`.
            num_proc (`int`, *optional*, defaults to `None`):
                Max number of processes when generating cache. Already cached shards are loaded sequentially.
            suffix_template (`str`):
                If `cache_file_name` is specified, then this suffix
                will be added at the end of the base name of each. Defaults to `"_{rank:05d}_of_{num_proc:05d}"`. For example, if `cache_file_name` is "processed.arrow", then for
                `rank=1` and `num_proc=4`, the resulting file would be `"processed_00001_of_00004.arrow"` for the default suffix.
            new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.
            desc (`str`, *optional*, defaults to `None`):
                Meaningful description to be displayed alongside with the progress bar while mapping examples.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> def add_prefix(example):
        ...     example["text"] = "Review: " + example["text"]
        ...     return example
        >>> ds = ds.map(add_prefix)
        >>> ds[0:3]["text"]
        ['Review: compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children .',
         'Review: the soundtrack alone is worth the price of admission .',
         'Review: rodriguez does a splendid job of racial profiling hollywood style--casting excellent latin actors of all ages--a trend long overdue .']

        # process a batch of examples
        >>> ds = ds.map(lambda example: tokenizer(example["text"]), batched=True)
        # set number of processors
        >>> ds = ds.map(add_prefix, num_proc=4)
        ```
        """
        if keep_in_memory and cache_file_name is not None:
            raise ValueError("Please use either `keep_in_memory` or `cache_file_name` but not both.")

        if num_proc is not None and num_proc <= 0:
            raise ValueError("num_proc must be an integer > 0.")

        # If the array is empty we do nothing (but we make sure to handle an empty indices mapping and remove the requested columns anyway)
        if self.num_rows== 0:
            return self

        if function is None:
            function = lambda x: x  # noqa: E731

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        if input_columns is not None:
            for input_column in input_columns:
                if input_column not in self.column_names:
                    raise ValueError(
                        f"Input column {input_column} not in the dataset. Current columns in the dataset: {self.column_names}"
                    )

        load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else True

        if fn_kwargs is None:
            fn_kwargs = {}

        if num_proc is not None and num_proc > self.num_rows:
            num_proc = self.num_rows
            logger.warning(
                f"num_proc must be <= {self.num_rows}. Reducing num_proc to {num_proc} for dataset of size {self.num_rows}."
            )

        disable_tqdm = not logging.is_progress_bar_enabled()

        if num_proc is None or num_proc == 1:
            return self._map_single(
                function=function,
                with_indices=with_indices,
                with_rank=with_rank,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=remove_columns,
                keep_in_memory=keep_in_memory,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=cache_file_name,
                writer_batch_size=writer_batch_size,
                features=features,
                disable_nullable=disable_nullable,
                fn_kwargs=fn_kwargs,
                new_fingerprint=new_fingerprint,
                disable_tqdm=disable_tqdm,
                desc=desc,
            )
        else:

            def format_cache_file_name(cache_file_name, rank):
                sep = cache_file_name.rindex(".")
                base_name, extension = cache_file_name[:sep], cache_file_name[sep:]
                cache_file_name = base_name + suffix_template.format(rank=rank, num_proc=num_proc) + extension
                logger.info(f"Process #{rank} will write at {cache_file_name}")
                return cache_file_name

            def format_new_fingerprint(new_fingerprint, rank):
                return new_fingerprint + suffix_template.format(rank=rank, num_proc=num_proc)

            prev_env = copy.deepcopy(os.environ)
            # check if parallelism if off
            # from https://github.com/huggingface/tokenizers/blob/bb668bc439dc34389b71dbb8ce0c597f15707b53/tokenizers/src/utils/parallelism.rs#L22
            if prev_env.get("TOKENIZERS_PARALLELISM", "false").lower() not in (
                "",
                "off",
                "false",
                "f",
                "no",
                "n",
                "0",
            ):
                logger.warning("Setting TOKENIZERS_PARALLELISM=false for forked processes.")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            initargs, initializer = None, None
            if not disable_tqdm:
                initargs, initializer = (RLock(),), tqdm.set_lock

            shards = [
                self.shard(num_shards=num_proc, index=rank)
                for rank in range(num_proc)
            ]
            kwds_per_shard = [
                dict(
                    self=shards[rank],
                    function=function,
                    with_indices=with_indices,
                    with_rank=with_rank,
                    input_columns=input_columns,
                    batched=batched,
                    batch_size=batch_size,
                    drop_last_batch=drop_last_batch,
                    remove_columns=remove_columns,
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    cache_file_name=format_cache_file_name(cache_file_name, rank)
                    if cache_file_name is not None
                    else None,
                    writer_batch_size=writer_batch_size,
                    features=features.copy() if features is not None else None,
                    disable_nullable=disable_nullable,
                    fn_kwargs=fn_kwargs,
                    rank=rank,
                    offset=sum(len(s) for s in shards[:rank]),
                    disable_tqdm=disable_tqdm,
                    new_fingerprint=format_new_fingerprint(new_fingerprint, rank)
                    if new_fingerprint is not None
                    else None,
                    desc=desc,
                )
                for rank in range(num_proc)
            ]

            # We search for already cached shards
            def catch_non_existent_error(func, kwargs):
                try:
                    return func(**kwargs)
                except NonExistentDatasetError:
                    return None

            transformed_shards = [
                catch_non_existent_error(self.__class__._map_single, dict(cache_only=True, **kwds))
                for kwds in kwds_per_shard
            ]

            # We try to create a pool with as many workers as dataset not yet cached.
            nb_of_missing_shards = transformed_shards.count(None)
            if nb_of_missing_shards > 0:
                with Pool(nb_of_missing_shards, initargs=initargs, initializer=initializer) as pool:
                    os.environ = prev_env
                    logger.info(f"Spawning {num_proc} processes")
                    results = {
                        i: pool.apply_async(self.__class__._map_single, kwds=kwds)
                        for i, (kwds, cached_shard) in enumerate(zip(kwds_per_shard, transformed_shards))
                        if cached_shard is None
                    }
                    assert (
                        len(results) == nb_of_missing_shards
                    ), "The number of missing cached shards needs to correspond to the number of `_map_single` we're running"

                    for index, async_result in results.items():
                        transformed_shards[index] = async_result.get()

            assert (
                transformed_shards.count(None) == 0
            ), "All shards have to be defined Datasets, none should still be missing."

            logger.info(f"Concatenating {num_proc} shards")

            # result = _concatenate_map_style_datasets(transformed_shards)
            # if new_fingerprint is not None:
            #     result._fingerprint = new_fingerprint
            # return result

    # @pysnooper.snoop()
    def _map_single(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[List[str]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[List[str]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        new_fingerprint: Optional[str] = None,
        rank: Optional[int] = None,
        offset: int = 0,
        disable_tqdm: bool = False,
        desc: Optional[str] = None,
        cache_only: bool = False,
    ) -> "Table":
        """Apply a function to all the elements in the table (individually or in batches)
        and update the table (if function does update examples).

        Args:
            function (`Callable`): with one of the following signature:
                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False` and `with_rank=False`
                - `function(example: Dict[str, Any], *extra_args) -> Dict[str, Any]` if `batched=False` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)
                - `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False` and `with_rank=False`
                - `function(batch: Dict[str, List], *extra_args) -> Dict[str, List]` if `batched=True` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)

                For advanced usage, the function can also return a `pyarrow.Table`.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the dataset unchanged.
                If no function is provided, default to identity function: lambda x: x
            with_indices (`bool`, defaults to `False`): Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx[, rank]): ...`.
            with_rank (`bool`, default `False`): Provide process rank to `function`. Note that in this case the signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`Optional[List[str]]`, defaults to `None`): The columns to be passed into `function` as
                positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`): Provide batch of examples to `function`
            batch_size (`int`, optional, defaults to `1000`): Number of examples per batch provided to `function` if `batched=True`
                `batch_size <= 0` or `batch_size == None`: Provide the full dataset as a single batch to `function`
            drop_last_batch (`bool`, default: `False`): Whether a last batch smaller than the batch_size should be
                dropped instead of being processed by the function.
            remove_columns (`Optional[List[str]]`, defaults to `None`): Remove a selection of columns while doing the mapping.
                Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
                columns with names in `remove_columns`, these columns will be kept.
            keep_in_memory (`bool`, defaults to `False`): Keep the dataset in memory instead of writing it to a cache file.
            load_from_cache_file (`bool`, defaults to `True` if caching is enabled): If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            cache_file_name (`str`, optional, defaults to `None`): Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
            writer_batch_size (`int`, default `1000`): Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `.map()`.
            features (`Optional[datasets.Features]`, defaults to `None`): Use a specific Features to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`): Disallow null values in the table.
            fn_kwargs (`Dict`, optional, defaults to `None`): Keyword arguments to be passed to `function`
            new_fingerprint (`str`, optional, defaults to `None`): the new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
            rank: (`int`, optional, defaults to `None`): If specified, this is the process rank when doing multiprocessing
            offset: (`int`, defaults to 0): If specified, this is an offset applied to the indices passed to `function` if `with_indices=True`.
            disable_tqdm (`bool`, defaults to `False`): Whether to silence tqdm's output.
            desc (`str`, optional, defaults to `None`): Meaningful description to be displayed alongside with the progress bar while mapping examples.
            cache_only (`bool`, defaults to `False`): Flag in order to notifiy the method will either find a cached dataset or raise `NonExistentDatasetError` exception,
        """

        from .formatting import format_table, get_formatter

        # Reduce logging to keep things readable in multiprocessing with tqdm
        if rank is not None and logging.get_verbosity() < logging.WARNING:
            logging.set_verbosity_warning()
        # Print at least one thing to fix tqdm in notebooks in multiprocessing
        # see https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308
        if rank is not None and not disable_tqdm and any("notebook" in tqdm_cls.__name__ for tqdm_cls in tqdm.__mro__):
            print(" ", end="", flush=True)

        if fn_kwargs is None:
            fn_kwargs = {}

        # If we do batch computation but no batch size is provided, default to the full dataset
        if batched and (batch_size is None or batch_size <= 0):
            batch_size = self.num_rows

        # Raise an error if we were supposed to return a cached dataset and none was found
        if cache_only:
            raise NonExistentDatasetError

        # We set this variable to True after processing the first example/batch in
        # `apply_function_on_filtered_inputs` if the map function returns a dict.
        # If set to False, no new arrow table will be created

        update_data = None

        self._format_kwargs={}
        self._format_type=None

        format_kwargs = self._format_kwargs.copy()
        # Lazy formatting is only available for the default format (None/python)
        if not input_columns and self._format_type is None:
            format_kwargs["lazy"] = True
        input_formatter = get_formatter(
            self._format_type,
            features=self.features,
            **format_kwargs,
        )

        class NumExamplesMismatchError(Exception):
            pass

        def validate_function_output(processed_inputs, indices):
            """Validate output of the map function."""
            if processed_inputs is not None and not isinstance(processed_inputs, (Mapping, pa.Table)):
                raise TypeError(
                    f"Provided `function` which is applied to all elements of table returns a variable of type {type(processed_inputs)}. Make sure provided `function` returns a variable of type `dict` (or a pyarrow table) to update the dataset or `None` if you are only interested in side effects."
                )
            elif isinstance(indices, list) and isinstance(processed_inputs, Mapping):
                allowed_batch_return_types = (list, np.ndarray, pd.Series)
                if config.TF_AVAILABLE and "tensorflow" in sys.modules:
                    import tensorflow as tf

                    allowed_batch_return_types += (tf.Tensor,)
                if config.TORCH_AVAILABLE and "torch" in sys.modules:
                    import torch

                    allowed_batch_return_types += (torch.Tensor,)
                if config.JAX_AVAILABLE and "jax" in sys.modules:
                    import jax.numpy as jnp

                    allowed_batch_return_types += (jnp.ndarray,)
                all_dict_values_are_lists = all(
                    isinstance(value, allowed_batch_return_types) for value in processed_inputs.values()
                )
                if all_dict_values_are_lists is False:
                    raise TypeError(
                        f"Provided `function` which is applied to all elements of table returns a `dict` of types {[type(x) for x in processed_inputs.values()]}. When using `batched=True`, make sure provided `function` returns a `dict` of types like `{allowed_batch_return_types}`."
                    )

        def apply_function_on_filtered_inputs(pa_inputs, indices, check_same_num_examples=False, offset=0):
            """Utility to apply the function on a selection of columns."""
            nonlocal update_data
            from .formatting import format_table, get_formatter, query_table
            from .formatting.formatting import LazyDict, _is_range_contiguous


            inputs = format_table(
                pa_inputs,
                0 if not batched else range(pa_inputs.num_rows),
                format_columns=input_columns,
                formatter=input_formatter,
            )
            fn_args = [inputs] if input_columns is None else [inputs[col] for col in input_columns]
            if offset == 0:
                effective_indices = indices
            else:
                effective_indices = [i + offset for i in indices] if isinstance(indices, list) else indices + offset
            additional_args = ()
            if with_indices:
                additional_args += (effective_indices,)
            if with_rank:
                additional_args += (rank,)
            processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
            if isinstance(processed_inputs, LazyDict):
                processed_inputs = {
                    k: v for k, v in processed_inputs.data.items() if k not in processed_inputs.keys_to_format
                }
                returned_lazy_dict = True
            else:
                returned_lazy_dict = False
            if update_data is None:
                # Check if the function returns updated examples
                update_data = isinstance(processed_inputs, (Mapping, pa.Table))
                validate_function_output(processed_inputs, indices)
            if not update_data:
                return None  # Nothing to update, let's move on
            if self._format_type or input_columns:
                # TODO(QL, MS): ideally the behavior should be the same even if the dataset is formatted (may require major release)
                inputs_to_merge = {k: v for k, v in zip(pa_inputs.column_names, pa_inputs.itercolumns())}
            elif isinstance(inputs, LazyDict):
                inputs_to_merge = {
                    k: (v if k not in inputs.keys_to_format else pa_inputs[k]) for k, v in inputs.data.items()
                }
            else:
                inputs_to_merge = inputs
            if remove_columns is not None:
                for column in remove_columns:
                    # `function` can modify input in-place causing column to be already removed.
                    if column in inputs_to_merge:
                        inputs_to_merge.pop(column)
                    if returned_lazy_dict and column in processed_inputs:
                        processed_inputs.pop(column)
            if check_same_num_examples:
                input_num_examples = len(pa_inputs)
                processed_inputs_num_examples = len(processed_inputs[next(iter(processed_inputs.keys()))])
                if input_num_examples != processed_inputs_num_examples:
                    raise NumExamplesMismatchError()
            if isinstance(inputs, Mapping) and isinstance(processed_inputs, Mapping):
                # The .map() transform *updates* the dataset:
                # the output dictionary contains both the the input data and the output data.
                # The output dictionary may contain Arrow values from `inputs_to_merge` so that we can re-write them efficiently.
                return {**inputs_to_merge, **processed_inputs}
            else:
                return processed_inputs
        # @pysnooper.snoop()
        def init_buffer_and_writer():
            from .arrow_writer import ArrowWriter
            # Prepare output buffer and batched writer in memory or on file if we update the table
            writer_features = features
            if writer_features is None:
                writer_features = self.features
                update_features = True
            else:
                update_features = False
            if keep_in_memory or cache_file_name is None:
                buf_writer = pa.BufferOutputStream()
                tmp_file = None
                writer = ArrowWriter(
                    features=writer_features,
                    stream=buf_writer,
                    writer_batch_size=writer_batch_size,
                    update_features=update_features,
                    fingerprint=new_fingerprint,
                    disable_nullable=disable_nullable,
                )
            else:
                buf_writer = None
                logger.info(f"Caching processed dataset at {cache_file_name}")
                tmp_file = tempfile.NamedTemporaryFile("wb", dir=os.path.dirname(cache_file_name), delete=False)
                writer = ArrowWriter(
                    features=writer_features,
                    path=tmp_file.name,
                    writer_batch_size=writer_batch_size,
                    update_features=update_features,
                    fingerprint=new_fingerprint,
                    disable_nullable=disable_nullable,
                )
            return buf_writer, writer, tmp_file

        # If `update_data` is True after processing the first example/batch, initalize these resources with `init_buffer_and_writer`
        buf_writer, writer, tmp_file = None, None, None

        # Optionally initialize the writer as a context manager
        with contextlib.ExitStack() as stack:
            try:
                # input_dataset = self.with_format("arrow")

                input_dataset=self
                input_dataset._format_type = 'arrow'

                # Loop over single examples or batches and write to buffer/file if examples are to be updated
                if not batched:
                    pbar_total = len(input_dataset)
                    pbar_iterable = enumerate(input_dataset)
                else:
                    num_rows = (
                        len(input_dataset) if not drop_last_batch else len(input_dataset) // batch_size * batch_size
                    )
                    pbar_total = (num_rows // batch_size) + 1 if num_rows % batch_size else num_rows // batch_size
                    pbar_iterable = zip(
                        range(0, num_rows, batch_size),
                        input_dataset.iter(batch_size, drop_last_batch=drop_last_batch),
                    )
                pbar_unit = "ex" if not batched else "ba"
                pbar_desc = (desc + " " if desc is not None else "") + "#" + str(rank) if rank is not None else desc
                pbar = logging.tqdm(
                    pbar_iterable,
                    total=pbar_total,
                    disable=disable_tqdm,
                    position=rank,
                    unit=pbar_unit,
                    desc=pbar_desc,
                )
                if not batched:
                    # 携带table
                    for i, example in pbar:
                        # from datasets.arrow_dataset import Dataset
                        # print(example)
                        example = apply_function_on_filtered_inputs(example, i, offset=offset)
                        # print(type(example))
                        # print(example)
                        if update_data:
                            if i == 0:
                                buf_writer, writer, tmp_file = init_buffer_and_writer()
                                stack.enter_context(writer)
                            if isinstance(example, pa.Table):
                                writer.write_row(example)
                            else:
                                writer.write(example)
                else:
                    for i, batch in pbar:
                        indices = list(
                            range(*(slice(i, i + batch_size).indices(input_dataset.num_rows)))
                        )  # Something simpler?
                        try:
                            batch = apply_function_on_filtered_inputs(
                                batch,
                                indices,
                                check_same_num_examples=len(input_dataset.list_indexes()) > 0,
                                offset=offset,
                            )
                        except NumExamplesMismatchError:
                            raise DatasetTransformationNotAllowedError(
                                "Using `.map` in batched mode on a dataset with attached indexes is allowed only if it doesn't create or remove existing examples. You can first run `.drop_index() to remove your index and then re-add it."
                            ) from None
                        if update_data:
                            if i == 0:
                                buf_writer, writer, tmp_file = init_buffer_and_writer()
                                stack.enter_context(writer)
                            if isinstance(batch, pa.Table):
                                writer.write_table(batch)
                            else:
                                writer.write_batch(batch)
                if update_data and writer is not None:
                    writer.finalize()  # close_stream=bool(buf_writer is None))  # We only close if we are writing in a file
            except (Exception, KeyboardInterrupt):
                if update_data:
                    if writer is not None:
                        writer.finalize()
                    if tmp_file is not None:
                        tmp_file.close()
                        if os.path.exists(tmp_file.name):
                            os.remove(tmp_file.name)
                raise

        if update_data and tmp_file is not None:
            tmp_file.close()
            shutil.move(tmp_file.name, cache_file_name)
            umask = os.umask(0o666)
            os.umask(umask)
            os.chmod(cache_file_name, 0o666 & ~umask)

        if update_data:
            if buf_writer is None:
                return self.from_file(cache_file_name)
            else:
                return self.from_buffer(buf_writer.getvalue())
        else:
            return self


    def shard(self,num_shards: int,index: int) -> "Table":
        if not 0 <= index < num_shards:
            raise ValueError("index should be in [0, num_shards-1]")

        div = len(self) // num_shards
        mod = len(self) % num_shards
        start = div * index + min(index, mod)
        end = start + div + (1 if index < mod else 0)

        return self.slice(start,end)


# The MemoryMappedTable needs replays to properly reload tables from the disk
Replay = Tuple[str, tuple, dict]


class MemoryMappedTable(TableBlock):
    """
    The table is said memory mapped when it doesn't use the user's RAM but loads the data
    from the disk instead.

    Pickling it doesn't copy the data into memory.
    Instead, only the path to the memory mapped arrow file is pickled, as well as the list
    of transforms to "replay" when reloading the table from the disk.

    Its implementation requires to store an history of all the transforms that were applied
    to the underlying pyarrow Table, so that they can be "replayed" when reloading the Table
    from the disk.

    This is different from the `InMemoryTable` table, for which pickling does copy all the
    data in memory.

    `InMemoryTable` must be used when data fit in memory, while `MemoryMapped` are reserved for
    data bigger than memory or when you want the memory footprint of your application to
    stay low.
    """

    def __init__(self, table: pa.Table, path: str, replays: Optional[List[Replay]] = None):
        super().__init__(table)
        self.path = os.path.abspath(path)
        self.replays: List[Replay] = replays if replays is not None else []

    @classmethod
    def from_file(cls, filename: str, replays=None):
        table = _memory_mapped_arrow_table_from_file(filename)
        table = cls._apply_replays(table, replays)
        return cls(table, filename, replays)

    def __getstate__(self):
        return {"path": self.path, "replays": self.replays}

    def __setstate__(self, state):
        path = state["path"]
        replays = state["replays"]
        table = _memory_mapped_arrow_table_from_file(path)
        table = self._apply_replays(table, replays)
        MemoryMappedTable.__init__(self, table, path=path, replays=replays)

    @staticmethod
    def _apply_replays(table: pa.Table, replays: Optional[List[Replay]] = None) -> pa.Table:
        if replays is not None:
            for name, args, kwargs in replays:
                if name == "cast":
                    table = table_cast(table, *args, **kwargs)
                elif name == "flatten":
                    table = table_flatten(table, *args, **kwargs)
                else:
                    table = getattr(table, name)(*args, **kwargs)
        return table

    def _append_replay(self, replay: Replay) -> List[Replay]:
        replays = copy.deepcopy(self.replays)
        replays.append(replay)
        return replays

    def slice(self, offset=0, length=None):
        """
        Compute zero-copy slice of this Table.

        Args:
            offset (`int`, defaults to `0`):
                Offset from start of table to slice.
            length (`int`, defaults to `None`):
                Length of slice (default is until end of table starting from
                offset).

        Returns:
            `datasets.table.Table`
        """
        replay = ("slice", (offset, length), {})
        replays = self._append_replay(replay)
        # Use fast slicing here
        return MemoryMappedTable(self.fast_slice(offset=offset, length=length), self.path, replays)

    def filter(self, *args, **kwargs):
        """
        Select records from a Table. See `pyarrow.compute.filter` for full usage.
        """
        replay = ("filter", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.filter(*args, **kwargs), self.path, replays)

    def flatten(self, *args, **kwargs):
        """
        Flatten this Table.  Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        replay = ("flatten", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(table_flatten(self.table, *args, **kwargs), self.path, replays)

    def combine_chunks(self, *args, **kwargs):
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the ChunkedArray of each column are
        concatenated into zero or one chunk.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        replay = ("combine_chunks", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.combine_chunks(*args, **kwargs), self.path, replays)

    def cast(self, *args, **kwargs):
        """
        Cast table values to another schema

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        replay = ("cast", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(table_cast(self.table, *args, **kwargs), self.path, replays)

    def replace_schema_metadata(self, *args, **kwargs):
        """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be None,
        which deletes any existing metadata.

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
        replay = ("replace_schema_metadata", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.replace_schema_metadata(*args, **kwargs), self.path, replays)

    def add_column(self, *args, **kwargs):
        """
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column added.
        """
        replay = ("add_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.add_column(*args, **kwargs), self.path, replays)

    def append_column(self, *args, **kwargs):
        """
        Append column at end of columns.

        Args:
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column added.
        """
        replay = ("append_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.append_column(*args, **kwargs), self.path, replays)

    def remove_column(self, *args, **kwargs):
        """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`:
                New table without the column.
        """
        replay = ("remove_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.remove_column(*args, **kwargs), self.path, replays)

    def set_column(self, *args, **kwargs):
        """
        Replace column in Table at position.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column set.
        """
        replay = ("set_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.set_column(*args, **kwargs), self.path, replays)

    def rename_columns(self, *args, **kwargs):
        """
        Create new table with columns renamed to provided names.
        """
        replay = ("rename_columns", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.rename_columns(*args, **kwargs), self.path, replays)

    def drop(self, *args, **kwargs):
        """
        Drop one or more columns and return a new table.

        Args:
            columns (`List[str]`):
                List of field names referencing existing columns.

        Raises:
            `KeyError` : if any of the passed columns name are not existing.

        Returns:
            `datasets.table.Table`:
                New table without the columns.
        """
        replay = ("drop", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.drop(*args, **kwargs), self.path, replays)

    def select(self, *args, **kwargs):
        """
        Select columns of the table.

        Returns a new table with the specified columns, and metadata preserved.

        Args:
            columns (:obj:`Union[List[str], List[int]]`):
                The column names or integer indices to select.

        Returns:
            :class:`datasets.table.Table`: New table with the specified columns, and metadata preserved.
        """
        replay = ("select", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.select(*args, **kwargs), self.path, replays)


# A ConcatenationTable is the concatenation of several tables.
# The ``blocks`` attributes stores a list of list of blocks.
# The first axis concatenates the tables along the axis 0 (it appends rows),
# while the second axis concatenates tables along the axis 1 (it appends columns).
TableBlockContainer = TypeVar("TableBlockContainer", TableBlock, List[TableBlock], List[List[TableBlock]])



def list_table_cache_files(table: Table) -> List[str]:
    """
    Get the cache files that are loaded by the table.
    Cache file are used when parts of the table come from the disk via memory mapping.

    Returns:
        `List[str]`:
            A list of paths to the cache files loaded by the table.
    """
    if isinstance(table, MemoryMappedTable):
        return [table.path]
    else:
        return []


def _wrap_for_chunked_arrays(func):
    """Apply the function on each chunk of a `pyarrow.ChunkedArray`, or on the array directly"""

    def wrapper(array, *args, **kwargs):
        if isinstance(array, pa.ChunkedArray):
            return pa.chunked_array([func(chunk, *args, **kwargs) for chunk in array.chunks])
        else:
            return func(array, *args, **kwargs)

    return wrapper


def _is_extension_type(pa_type: pa.DataType) -> bool:
    """
    Check (recursively) if a pyarrow type is an extension type.
    """
    if isinstance(pa_type, pa.StructType):
        return any(_is_extension_type(field.type) for field in pa_type)
    elif isinstance(pa_type, (pa.ListType, pa.FixedSizeListType, pa.LargeListType)):
        return _is_extension_type(pa_type.value_type)
    elif isinstance(pa_type, pa.ExtensionType):
        return True
    else:
        return False


def array_concat(arrays: List[pa.Array]):
    """Improved version of pa.concat_arrays

    It supports concatenating pa.ExtensionArray objects by concatenating the underlying storages.

    Args:
        arrays (List[pa.Array]): List of arrays to contatenate

    Raises:
        pa.ArrowInvalid: if the arrow array concatenation fails
        ValueError: if the list of arrays is empty
        TypeError: if the arrays to be concatenated have different types

    Returns:
        array (:obj:`pyarrow.Array`): the concatenated array
    """
    arrays = list(arrays)
    array_types = set(array.type for array in arrays)

    if not array_types:
        raise ValueError("Couldn't concatenate empty list of arrays")
    if len(array_types) > 1:
        array_types = list(array_types)
        raise TypeError(f"Couldn't concatenate arrays with different types {array_types[0]} and {array_types[1]}")

    array_type = arrays[0].type
    arrays = [chunk for arr in arrays for chunk in (arr.chunks if isinstance(arr, pa.ChunkedArray) else (arr,))]

    if not _is_extension_type(array_type):
        return pa.concat_arrays(arrays)

    def _offsets_concat(offsets):
        offset = offsets[0]
        concatenated_offsets = offset
        for offset in offsets[1:]:
            offset = pc.subtract(offset, offset[0])
            offset = pc.add(offset[1:], concatenated_offsets[-1])
            concatenated_offsets = pa.concat_arrays([concatenated_offsets, offset])
        return concatenated_offsets

    def _concat_arrays(arrays):
        array_type = arrays[0].type
        if isinstance(array_type, pa.PyExtensionType):
            return array_type.wrap_array(_concat_arrays([array.storage for array in arrays]))
        elif pa.types.is_struct(array_type):
            return pa.StructArray.from_arrays(
                [_concat_arrays([array.field(field.name) for array in arrays]) for field in array_type],
                fields=list(array_type),
                mask=pa.concat_arrays([array.is_null() for array in arrays]),
            )
        elif pa.types.is_list(array_type):
            if any(array.null_count > 0 for array in arrays):
                if config.PYARROW_VERSION.major < 10:
                    warnings.warn(
                        "None values are converted to empty lists in `pyarrow<10.0.0` when concatenating list arrays with None values. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676."
                    )
                else:
                    return pa.ListArray.from_arrays(
                        _offsets_concat([array.offsets for array in arrays]),
                        _concat_arrays([array.values for array in arrays]),
                        mask=pa.concat_arrays([array.is_null() for array in arrays]),
                    )
            return pa.ListArray.from_arrays(
                _offsets_concat([array.offsets for array in arrays]),
                _concat_arrays([array.values for array in arrays]),
            )
        elif pa.types.is_fixed_size_list(array_type):
            return pa.FixedSizeListArray.from_arrays(
                _concat_arrays([array.values for array in arrays]),
                array_type.list_size,
            )
        return pa.concat_arrays(arrays)

    return _concat_arrays(arrays)

# 把pyarrow 数组转换新的格式
@_wrap_for_chunked_arrays
# @pysnooper.snoop()
def array_cast(array: pa.Array, pa_type: pa.DataType, allow_number_to_str=True):
    # print(pa_type)
    """Improved version of `pa.Array.cast`

    It supports casting `pa.StructArray` objects to re-order the fields.
    It also let you control certain aspects of the casting, e.g. whether
    to disable numbers (`floats` or `ints`) to strings.

    Args:
        array (`pa.Array`):
            PyArrow array to cast
        pa_type (`pa.DataType`):
            Target PyArrow type
        allow_number_to_str (`bool`, defaults to `True`):
            Whether to allow casting numbers to strings.
            Defaults to `True`.

    Raises:
        `pa.ArrowInvalidError`: if the arrow data casting fails
        `TypeError`: if the target type is not supported according, e.g.

            - if a field is missing
            - if casting from numbers to strings and `allow_number_to_str` is `False`

    Returns:
        `List[pyarrow.Array]`: the casted array
    """
    _c = partial(array_cast, allow_number_to_str=allow_number_to_str)
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if isinstance(pa_type, pa.ExtensionType):
        return pa_type.wrap_array(array)
    elif array.type == pa_type:
        return array
    elif pa.types.is_struct(array.type):
        if pa.types.is_struct(pa_type) and (
            set(field.name for field in pa_type) == set(field.name for field in array.type)
        ):
            arrays = [_c(array.field(field.name), field.type) for field in pa_type]
            return pa.StructArray.from_arrays(arrays, fields=list(pa_type), mask=array.is_null())
    elif pa.types.is_list(array.type):
        if pa.types.is_fixed_size_list(pa_type):
            if pa_type.list_size * len(array) == len(array.values):
                return pa.FixedSizeListArray.from_arrays(
                    _c(array.values, pa_type.value_type),
                    pa_type.list_size,
                )
        elif pa.types.is_list(pa_type):
            if array.null_count > 0:
                if config.PYARROW_VERSION.major < 10:
                    warnings.warn(
                        f"None values are converted to empty lists in `pyarrow<10.0.0` when converting array to {pa_type}. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676."
                    )
                else:
                    return pa.ListArray.from_arrays(
                        array.offsets, _c(array.values, pa_type.value_type), mask=array.is_null()
                    )
            return pa.ListArray.from_arrays(array.offsets, _c(array.values, pa_type.value_type))
    elif pa.types.is_fixed_size_list(array.type):
        if pa.types.is_fixed_size_list(pa_type):
            return pa.FixedSizeListArray.from_arrays(
                _c(array.values, pa_type.value_type),
                pa_type.list_size,
            )
        elif pa.types.is_list(pa_type):
            offsets_arr = pa.array(range(len(array) + 1), pa.int32())
            if array.null_count > 0:
                if config.PYARROW_VERSION.major < 10:
                    warnings.warn(
                        f"None values are converted to empty lists in `pyarrow<10.0.0` when converting array to {pa_type}. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676."
                    )
                else:
                    return pa.ListArray.from_arrays(
                        offsets_arr, _c(array.values, pa_type.value_type), mask=array.is_null()
                    )
            return pa.ListArray.from_arrays(offsets_arr, _c(array.values, pa_type.value_type))
    else:
        if (
            not allow_number_to_str
            and pa.types.is_string(pa_type)
            and (pa.types.is_floating(array.type) or pa.types.is_integer(array.type))
        ):
            raise TypeError(
                f"Couldn't cast array of type {array.type} to {pa_type} since allow_number_to_str is set to {allow_number_to_str}"
            )
        if pa.types.is_null(pa_type) and not pa.types.is_null(array.type):
            raise TypeError(f"Couldn't cast array of type {array.type} to {pa_type}")
        return array.cast(pa_type)
    raise TypeError(f"Couldn't cast array of type\n{array.type}\nto\n{pa_type}")

# 把pyarrow数据转换为特征数组
@_wrap_for_chunked_arrays
# @pysnooper.snoop()
def cast_array_to_feature(array: pa.Array, feature: "FeatureType", allow_number_to_str=True):
    # print(feature())
    """Cast an array to the arrow type that corresponds to the requested feature type.
    For custom features like [`Audio`] or [`Image`], it takes into account the "cast_storage" methods
    they defined to enable casting from other arrow types.

    Args:
        array (`pa.Array`):
            The PyArrow array to cast.
        feature (`datasets.features.FeatureType`):
            The target feature type.
        allow_number_to_str (`bool`, defaults to `True`):
            Whether to allow casting numbers to strings.
            Defaults to `True`.

    Raises:
        `pa.ArrowInvalidError`: if the arrow data casting fails
        `TypeError`: if the target type is not supported according, e.g.

            - if a field is missing
            - if casting from numbers to strings and `allow_number_to_str` is `False`

    Returns:
        array (`pyarrow.Array`): the casted array
    """
    from .features.features import Sequence, get_nested_type

    _c = partial(cast_array_to_feature, allow_number_to_str=allow_number_to_str)

    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if hasattr(feature, "cast_storage"):
        return feature.cast_storage(array)
    elif pa.types.is_struct(array.type):
        # feature must be a dict or Sequence(subfeatures_dict)
        if isinstance(feature, Sequence) and isinstance(feature.feature, dict):
            feature = {
                name: Sequence(subfeature, length=feature.length) for name, subfeature in feature.feature.items()
            }
        if isinstance(feature, dict) and set(field.name for field in array.type) == set(feature):
            arrays = [_c(array.field(name), subfeature) for name, subfeature in feature.items()]
            return pa.StructArray.from_arrays(arrays, names=list(feature), mask=array.is_null())
    elif pa.types.is_list(array.type):
        # feature must be either [subfeature] or Sequence(subfeature)
        if isinstance(feature, list):
            casted_values = _c(array.values, feature[0])
            if casted_values.type == array.values.type:
                return array
            else:
                if array.null_count > 0:
                    if config.PYARROW_VERSION.major < 10:
                        warnings.warn(
                            f"None values are converted to empty lists in `pyarrow<10.0.0` when converting array to {feature}. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676."
                        )
                    else:
                        return pa.ListArray.from_arrays(array.offsets, casted_values, mask=array.is_null())
                return pa.ListArray.from_arrays(array.offsets, casted_values)
        elif isinstance(feature, Sequence):
            if feature.length > -1:
                if feature.length * len(array) == len(array.values):
                    return pa.FixedSizeListArray.from_arrays(_c(array.values, feature.feature), feature.length)
            else:
                casted_values = _c(array.values, feature.feature)
                if casted_values.type == array.values.type:
                    return array
                else:
                    if array.null_count > 0:
                        if config.PYARROW_VERSION.major < 10:
                            warnings.warn(
                                f"None values are converted to empty lists in `pyarrow<10.0.0` when converting array to {feature}. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676."
                            )
                        else:
                            return pa.ListArray.from_arrays(
                                array.offsets, _c(array.values, feature.feature), mask=array.is_null()
                            )
                    return pa.ListArray.from_arrays(array.offsets, _c(array.values, feature.feature))
    elif pa.types.is_fixed_size_list(array.type):
        # feature must be either [subfeature] or Sequence(subfeature)
        if isinstance(feature, list):
            if array.null_count > 0:
                if config.PYARROW_VERSION.major < 10:
                    warnings.warn(
                        f"None values are converted to empty lists when converting array to {feature}. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676. This will raise an error in a future major version of `datasets`"
                    )
                else:
                    return pa.ListArray.from_arrays(array.offsets, _c(array.values, feature[0]), mask=array.is_null())
            return pa.ListArray.from_arrays(array.offsets, _c(array.values, feature[0]))
        elif isinstance(feature, Sequence):
            if feature.length > -1:
                if feature.length * len(array) == len(array.values):
                    return pa.FixedSizeListArray.from_arrays(_c(array.values, feature.feature), feature.length)
            else:
                offsets_arr = pa.array(range(len(array) + 1), pa.int32())
                if array.null_count > 0:
                    if config.PYARROW_VERSION.major < 10:
                        warnings.warn(
                            f"None values are converted to empty lists when converting array to {feature}. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676. This will raise an error in a future major version of `datasets`"
                        )
                    else:
                        return pa.ListArray.from_arrays(
                            offsets_arr, _c(array.values, feature.feature), mask=array.is_null()
                        )
                return pa.ListArray.from_arrays(offsets_arr, _c(array.values, feature.feature))
    if pa.types.is_null(array.type):
        return array_cast(array, get_nested_type(feature), allow_number_to_str=allow_number_to_str)

    elif not isinstance(feature, (Sequence, dict, list, tuple)):
        # 把数据转化为feature
        return array_cast(array, feature(), allow_number_to_str=allow_number_to_str)
    raise TypeError(f"Couldn't cast array of type\n{array.type}\nto\n{feature}")


@_wrap_for_chunked_arrays
def embed_array_storage(array: pa.Array, feature: "FeatureType"):
    """Embed data into an arrays's storage.
    For custom features like Audio or Image, it takes into account the "embed_storage" methods
    they defined to enable embedding external data (e.g. an image file) into an other arrow types.

    <Added version="2.4.0"/>

    Args:
        array (`pa.Array`):
            The PyArrow array in which to embed data.
        feature (`datasets.features.FeatureType`):
            Array features.

    Raises:
        `TypeError`: if the target type is not supported according, e.g.

            - if a field is missing

    Returns:
         array (`pyarrow.Array`): the casted array
    """
    from .features import Sequence

    _e = embed_array_storage

    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if hasattr(feature, "embed_storage"):
        return feature.embed_storage(array)
    elif pa.types.is_struct(array.type):
        # feature must be a dict or Sequence(subfeatures_dict)
        if isinstance(feature, Sequence) and isinstance(feature.feature, dict):
            feature = {
                name: Sequence(subfeature, length=feature.length) for name, subfeature in feature.feature.items()
            }
        if isinstance(feature, dict):
            arrays = [_e(array.field(name), subfeature) for name, subfeature in feature.items()]
            return pa.StructArray.from_arrays(arrays, names=list(feature), mask=array.is_null())
    elif pa.types.is_list(array.type):
        # feature must be either [subfeature] or Sequence(subfeature)
        if isinstance(feature, list):
            if array.null_count > 0:
                if config.PYARROW_VERSION.major < 10:
                    warnings.warn(
                        f"None values are converted to empty lists when embedding array storage with {feature}. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676. This will raise an error in a future major version of `datasets`"
                    )
                else:
                    return pa.ListArray.from_arrays(array.offsets, _e(array.values, feature[0]), mask=array.is_null())
            return pa.ListArray.from_arrays(array.offsets, _e(array.values, feature[0]))
        elif isinstance(feature, Sequence):
            if feature.length > -1:
                if feature.length * len(array) == len(array.values):
                    return pa.FixedSizeListArray.from_arrays(_e(array.values, feature.feature), feature.length)
            else:
                casted_values = _e(array.values, feature.feature)
                if casted_values.type == array.values.type:
                    return array
                else:
                    if array.null_count > 0:
                        if config.PYARROW_VERSION.major < 10:
                            warnings.warn(
                                f"None values are converted to empty lists when embedding array storage with {feature}. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676. This will raise an error in a future major version of `datasets`"
                            )
                        else:
                            return pa.ListArray.from_arrays(
                                array.offsets, _e(array.values, feature.feature), mask=array.is_null()
                            )
                    return pa.ListArray.from_arrays(array.offsets, _e(array.values, feature.feature))
    elif pa.types.is_fixed_size_list(array.type):
        # feature must be either [subfeature] or Sequence(subfeature)
        if isinstance(feature, list):
            if array.null_count > 0:
                if config.PYARROW_VERSION.major < 10:
                    warnings.warn(
                        f"None values are converted to empty lists when embedding array storage with {feature}. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676. This will raise an error in a future major version of `datasets`"
                    )
                else:
                    return pa.ListArray.from_arrays(array.offsets, _e(array.values, feature[0]), mask=array.is_null())
            return pa.ListArray.from_arrays(array.offsets, _e(array.values, feature[0]))
        elif isinstance(feature, Sequence):
            if feature.length > -1:
                if feature.length * len(array) == len(array.values):
                    return pa.FixedSizeListArray.from_arrays(_e(array.values, feature.feature), feature.length)
            else:
                offsets_arr = pa.array(range(len(array) + 1), pa.int32())
                if array.null_count > 0:
                    if config.PYARROW_VERSION.major < 10:
                        warnings.warn(
                            f"None values are converted to empty lists when embedding array storage with {feature}. Install `pyarrow>=10.0.0` to avoid this behavior. More info: https://github.com/huggingface/datasets/issues/3676. This will raise an error in a future major version of `datasets`"
                        )
                    else:
                        return pa.ListArray.from_arrays(
                            offsets_arr, _e(array.values, feature.feature), mask=array.is_null()
                        )
                return pa.ListArray.from_arrays(offsets_arr, _e(array.values, feature.feature))
    if not isinstance(feature, (Sequence, dict, list, tuple)):
        return array
    raise TypeError(f"Couldn't embed array of type\n{array.type}\nwith\n{feature}")


def cast_table_to_features(table: pa.Table, features: "Features"):
    """Cast a table to the arrow schema that corresponds to the requested features.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to cast.
        features ([`Features`]):
            Target features.

    Returns:
        table (`pyarrow.Table`): the casted table
    """
    if sorted(table.column_names) != sorted(features):
        raise ValueError(f"Couldn't cast\n{table.schema}\nto\n{features}\nbecause column names don't match")
    arrays = [cast_array_to_feature(table[name], feature) for name, feature in features.items()]
    return pa.Table.from_arrays(arrays, schema=features.arrow_schema)

# 转变table成新的schema
def cast_table_to_schema(table: pa.Table, schema: pa.Schema):
    """Cast a table to the arrow schema. Different from `cast_table_to_features`, this method can preserve nullability.

    Args:
        table (`pa.Table`):
            PyArrow table to cast.
        features ([`Features`]):
            Target features.

    Returns:
        `pa.Table`: the casted table
    """
    from .features import Features
    features = Features.from_arrow_schema(schema)   # 根据schema 转化为feature
    if sorted(table.column_names) != sorted(features):
        raise ValueError(f"Couldn't cast\n{table.schema}\nto\n{features}\nbecause column names don't match")
    arrays = [cast_array_to_feature(table[name], feature) for name, feature in features.items()]
    return pa.Table.from_arrays(arrays, schema=schema)


# 更新table的schema信息，主要是空白table的schema信息补齐
def update_metadata_with_features(table: Table, features):
    """To be used in dataset transforms that modify the features of the dataset, in order to update the features stored in the metadata of its schema."""
    features = Features({col_name: features[col_name] for col_name in table.column_names})
    # 空白table的时候
    if table.schema.metadata is None or b"cube-studio" not in table.schema.metadata:
        pa_metadata = {"cube-studio": json.dumps({"info":{"features":features.to_dict()}})}
    else:
        metadata = json.loads(table.schema.metadata[b"cube-studio"].decode())
        metadata["info"]["features"] = features.to_dict()
        pa_metadata = {"cube-studio": json.dumps(metadata)}
    table = table.replace_schema_metadata(pa_metadata)
    return table


def embed_table_storage(table: pa.Table):
    """Embed external data into a table's storage.

    <Added version="2.4.0"/>

    Args:
        table (`pyarrow.Table`):
            PyArrow table in which to embed data.

    Returns:
        table (`pyarrow.Table`): the table with embedded data
    """
    from .features.features import Features, require_storage_embed

    features = Features.from_arrow_schema(table.schema)
    arrays = [
        embed_array_storage(table[name], feature) if require_storage_embed(feature) else table[name]
        for name, feature in features.items()
    ]
    return pa.Table.from_arrays(arrays, schema=features.arrow_schema)


def table_cast(table: pa.Table, schema: pa.Schema):
    """Improved version of `pa.Table.cast`.

    It supports casting to feature types stored in the schema metadata.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to cast.
        schema (`pyarrow.Schema`):
            Target PyArrow schema.

    Returns:
        table (`pyarrow.Table`): the casted table
    """
    if table.schema != schema:
        return cast_table_to_schema(table, schema)
    elif table.schema.metadata != schema.metadata:
        return table.replace_schema_metadata(schema.metadata)
    else:
        return table


def table_flatten(table: pa.Table):
    """Improved version of `pa.Table.flatten`.

    It behaves as `pa.Table.flatten` in a sense it does 1-step flatten of the columns with a struct type into one column per struct field,
    but updates the metadata and skips decodable features unless the `decode` attribute of these features is set to False.

    Args:
        table (`pa.Table`):
            PyArrow table to flatten.

    Returns:
        `Table`: the flattened table
    """
    from .features import Features

    features = Features.from_arrow_schema(table.schema)
    if any(hasattr(subfeature, "flatten") and subfeature.flatten() == subfeature for subfeature in features.values()):
        flat_arrays = []
        flat_column_names = []
        for field in table.schema:
            array = table.column(field.name)
            subfeature = features[field.name]
            if pa.types.is_struct(field.type) and (
                not hasattr(subfeature, "flatten") or subfeature.flatten() != subfeature
            ):
                flat_arrays.extend(array.flatten())
                flat_column_names.extend([f"{field.name}.{subfield.name}" for subfield in field.type])
            else:
                flat_arrays.append(array)
                flat_column_names.append(field.name)
        flat_table = pa.Table.from_arrays(
            flat_arrays,
            names=flat_column_names,
        )
    else:
        flat_table = table.flatten()
    # Preserve complex types in the metadata
    flat_features = features.flatten(max_depth=2)
    flat_features = Features({column_name: flat_features[column_name] for column_name in flat_table.column_names})
    return flat_table.replace_schema_metadata(flat_features.arrow_schema.metadata)


def table_visitor(table: pa.Table, function: Callable[[pa.Array], None]):
    """Visit all arrays in a table and apply a function to them.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to visit.
        function (`Callable[[pa.Array], None]`):
            Function to apply to each array.
    """
    from .features import Features, Sequence

    features = Features.from_arrow_schema(table.schema)

    def _visit(array, feature):
        if isinstance(array, pa.ChunkedArray):
            for chunk in array.chunks:
                _visit(chunk, feature)
        else:
            if isinstance(array, pa.ExtensionArray):
                array = array.storage
            function(array, feature)
            if pa.types.is_struct(array.type) and not hasattr(feature, "cast_storage"):
                if isinstance(feature, Sequence) and isinstance(feature.feature, dict):
                    feature = {
                        name: Sequence(subfeature, length=feature.length)
                        for name, subfeature in feature.feature.items()
                    }
                for name, subfeature in feature.items():
                    _visit(array.field(name), subfeature)
            elif pa.types.is_list(array.type):
                if isinstance(feature, list):
                    _visit(array.values, feature[0])
                elif isinstance(feature, Sequence):
                    _visit(array.values, feature.feature)

    for name, feature in features.items():
        _visit(table[name], feature)


def table_iter(table: Table, batch_size: int, drop_last_batch=False) -> Iterator[pa.Table]:
    """Iterate over sub-tables of size `batch_size`.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to iterate over.
        batch_size (`int`):
            Size of each sub-table to yield.
        drop_last_batch (`bool`, defaults to `False`):
            Drop the last batch if it is smaller than `batch_size`.
    """
    chunks_buffer = []
    chunks_buffer_size = 0
    for chunk in table.to_reader(max_chunksize=batch_size):
        if len(chunk) == 0:
            continue
        elif chunks_buffer_size + len(chunk) < batch_size:
            chunks_buffer.append(chunk)
            chunks_buffer_size += len(chunk)
            continue
        elif chunks_buffer_size + len(chunk) == batch_size:
            chunks_buffer.append(chunk)
            yield pa.Table.from_batches(chunks_buffer)
            chunks_buffer = []
            chunks_buffer_size = 0
        else:
            cropped_chunk_length = batch_size - chunks_buffer_size
            chunks_buffer.append(chunk.slice(0, cropped_chunk_length))
            yield pa.Table.from_batches(chunks_buffer)
            chunks_buffer = [chunk.slice(cropped_chunk_length, len(chunk) - cropped_chunk_length)]
            chunks_buffer_size = len(chunk) - cropped_chunk_length
    if not drop_last_batch and chunks_buffer:
        yield pa.Table.from_batches(chunks_buffer)


class NonExistentDatasetError(Exception):
    """Used when we expect the existence of a dataset"""

    pass

class DatasetTransformationNotAllowedError(Exception):
    pass