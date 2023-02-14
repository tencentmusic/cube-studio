

import pyarrow as pa
from functools import partial
from . import config
import warnings


def _wrap_for_chunked_arrays(func):
    """Apply the function on each chunk of a `pyarrow.ChunkedArray`, or on the array directly"""

    def wrapper(array, *args, **kwargs):
        if isinstance(array, pa.ChunkedArray):
            return pa.chunked_array([func(chunk, *args, **kwargs) for chunk in array.chunks])
        else:
            return func(array, *args, **kwargs)

    return wrapper


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
