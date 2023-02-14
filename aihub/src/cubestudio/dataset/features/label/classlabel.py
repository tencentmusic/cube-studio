import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field, fields
from typing import Any, ClassVar, Dict, List, Optional
from typing import Sequence as Sequence_
from typing import Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from ...core import array_cast

@dataclass
class ClassLabel:
    """Feature type for integer class labels.

    There are 3 ways to define a `ClassLabel`, which correspond to the 3 arguments:

     * `num_classes`: Create 0 to (num_classes-1) labels.
     * `names`: List of label strings.
     * `names_file`: File containing the list of labels.

    Under the hood the labels are stored as integers.
    You can use negative integers to represent unknown/missing labels.

    Args:
        num_classes (`int`, *optional*):
            Number of classes. All labels must be < `num_classes`.
        names (`list` of `str`, *optional*):
            String names for the integer classes.
            The order in which the names are provided is kept.
        names_file (`str`, *optional*):
            Path to a file with names for the integer classes, one per line.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'label': ClassLabel(num_classes=3, names=['bad', 'ok', 'good'])})
    >>> features
    {'label': ClassLabel(num_classes=3, names=['bad', 'ok', 'good'], id=None)}
    ```
    """

    num_classes: InitVar[Optional[int]] = None  # Pseudo-field: ignored by asdict/fields when converting to/from dict
    names: List[str] = None
    names_file: InitVar[Optional[str]] = None  # Pseudo-field: ignored by asdict/fields when converting to/from dict
    id: Optional[str] = None
    # Automatically constructed
    dtype: ClassVar[str] = "int64"
    pa_type: ClassVar[Any] = pa.int64()
    _str2int: ClassVar[Dict[str, int]] = None
    _int2str: ClassVar[Dict[int, int]] = None
    _type: str = field(default="ClassLabel", init=False, repr=False)

    def __post_init__(self, num_classes, names_file):
        self.num_classes = num_classes
        self.names_file = names_file
        if self.names_file is not None and self.names is not None:
            raise ValueError("Please provide either names or names_file but not both.")
        # Set self.names
        if self.names is None:
            if self.names_file is not None:
                self.names = self._load_names_from_file(self.names_file)
            elif self.num_classes is not None:
                self.names = [str(i) for i in range(self.num_classes)]
            else:
                raise ValueError("Please provide either num_classes, names or names_file.")
        # Set self.num_classes
        if self.num_classes is None:
            self.num_classes = len(self.names)
        elif self.num_classes != len(self.names):
            raise ValueError(
                "ClassLabel number of names do not match the defined num_classes. "
                f"Got {len(self.names)} names VS {self.num_classes} num_classes"
            )
        # Prepare mappings
        self._int2str = [str(name) for name in self.names]
        self._str2int = {name: i for i, name in enumerate(self._int2str)}
        if len(self._int2str) != len(self._str2int):
            raise ValueError("Some label names are duplicated. Each label name should be unique.")

    def __call__(self):
        return self.pa_type

    def str2int(self, values: Union[str, Iterable]) -> Union[int, Iterable]:
        """Conversion class name `string` => `integer`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> ds.features["label"].str2int('neg')
        0
        ```
        """
        if not isinstance(values, str) and not isinstance(values, Iterable):
            raise ValueError(
                f"Values {values} should be a string or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
            )
        return_list = True
        if isinstance(values, str):
            values = [values]
            return_list = False

        output = [self._strval2int(value) for value in values]
        return output if return_list else output[0]

    def _strval2int(self, value: str) -> int:
        failed_parse = False
        value = str(value)
        # first attempt - raw string value
        int_value = self._str2int.get(value)
        if int_value is None:
            # second attempt - strip whitespace
            int_value = self._str2int.get(value.strip())
            if int_value is None:
                # third attempt - convert str to int
                try:
                    int_value = int(value)
                except ValueError:
                    failed_parse = True
                else:
                    if int_value < -1 or int_value >= self.num_classes:
                        failed_parse = True
        if failed_parse:
            raise ValueError(f"Invalid string class label {value}")
        return int_value

    def int2str(self, values: Union[int, Iterable]) -> Union[str, Iterable]:
        """Conversion `integer` => class name `string`.

        Regarding unknown/missing labels: passing negative integers raises `ValueError`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> ds.features["label"].int2str(0)
        'neg'
        ```
        """
        if not isinstance(values, int) and not isinstance(values, Iterable):
            raise ValueError(
                f"Values {values} should be an integer or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
            )
        return_list = True
        if isinstance(values, int):
            values = [values]
            return_list = False

        for v in values:
            if not 0 <= v < self.num_classes:
                raise ValueError(f"Invalid integer class label {v:d}")

        output = [self._int2str[int(v)] for v in values]
        return output if return_list else output[0]

    def encode_example(self, example_data):
        if self.num_classes is None:
            raise ValueError(
                "Trying to use ClassLabel feature with undefined number of class. "
                "Please set ClassLabel.names or num_classes."
            )

        # If a string is given, convert to associated integer
        if isinstance(example_data, str):
            example_data = self.str2int(example_data)

        # Allowing -1 to mean no label.
        if not -1 <= example_data < self.num_classes:
            raise ValueError(f"Class label {example_data:d} greater than configured num_classes {self.num_classes}")
        return example_data

    def cast_storage(self, storage: Union[pa.StringArray, pa.IntegerArray]) -> pa.Int64Array:
        """Cast an Arrow array to the `ClassLabel` arrow storage type.
        The Arrow types that can be converted to the `ClassLabel` pyarrow storage type are:

        - `pa.string()`
        - `pa.int()`

        Args:
            storage (`Union[pa.StringArray, pa.IntegerArray]`):
                PyArrow array to cast.

        Returns:
            `pa.Int64Array`: Array in the `ClassLabel` arrow storage type.
        """
        if isinstance(storage, pa.IntegerArray):
            min_max = pc.min_max(storage).as_py()
            if min_max["max"] >= self.num_classes:
                raise ValueError(
                    f"Class label {min_max['max']} greater than configured num_classes {self.num_classes}"
                )
        elif isinstance(storage, pa.StringArray):
            storage = pa.array(
                [self._strval2int(label) if label is not None else None for label in storage.to_pylist()]
            )
        return array_cast(storage, self.pa_type)

    @staticmethod
    def _load_names_from_file(names_filepath):
        with open(names_filepath, encoding="utf-8") as f:
            return [name.strip() for name in f.read().split("\n") if name.strip()]  # Filter empty names
