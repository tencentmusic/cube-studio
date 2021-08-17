from __future__ import unicode_literals

import sys
import warnings
from enum import Enum

import pysnooper
from marshmallow import ValidationError
from marshmallow.fields import Field

PY2 = sys.version_info.major == 2
# ugh Python 2
if PY2:
    string_types = (str, unicode)  # noqa: F821
    text_type = unicode  # noqa: F821
else:
    string_types = (str, )
    text_type = str


class LoadDumpOptions(Enum):
    value = 1
    name = 0


class EnumField(Field):
    VALUE = LoadDumpOptions.value
    NAME = LoadDumpOptions.name

    default_error_messages = {
        'by_name': 'Invalid enum member {input}',
        'by_value': 'Invalid enum value {input}',
        'must_be_string': 'Enum name must be string'
    }

    # @pysnooper.snoop()
    def __init__(
            self, enum, by_value=False, load_by=None, dump_by=None, error='', *args, **kwargs
    ):
        self.enum = enum
        self.by_value = by_value

        if error and any(old in error for old in ('name}', 'value}', 'choices}')):
            warnings.warn(
                "'name', 'value', and 'choices' fail inputs are deprecated,"
                "use input, names and values instead",
                DeprecationWarning,
                stacklevel=2
            )

        self.error = error

        if load_by is None:
            load_by = LoadDumpOptions.value if by_value else LoadDumpOptions.name

        if not isinstance(load_by, Enum) or load_by not in LoadDumpOptions:
            raise ValueError(
                'Invalid selection for load_by must be EnumField.VALUE or EnumField.NAME, got {}'.
                format(load_by)
            )

        if dump_by is None:
            dump_by = LoadDumpOptions.value if by_value else LoadDumpOptions.name

        if not isinstance(dump_by, Enum) or dump_by not in LoadDumpOptions:
            raise ValueError(
                'Invalid selection for load_by must be EnumField.VALUE or EnumField.NAME, got {}'.
                format(dump_by)
            )

        self.load_by = load_by
        self.dump_by = dump_by

        super(EnumField, self).__init__(*args, **kwargs)

    # def _serialize(self, value, attr, obj):
    #     if value is None:
    #         return None
    #     elif self.dump_by == LoadDumpOptions.value:
    #         return value.value
    #     else:
    #         return value.name

    # @pysnooper.snoop()
    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return None
        elif self.load_by == LoadDumpOptions.value:
            return self._deserialize_by_value(value, attr, data)
        else:
            return self._deserialize_by_name(value, attr, data)

    # @pysnooper.snoop()
    def _deserialize_by_value(self, value, attr, data):
        try:
            return value # self.enum(value)
        except ValueError:
            self.fail('by_value', input=value, value=value)

    # @pysnooper.snoop(watch_explode=('enum'))
    def _deserialize_by_name(self, value, attr, data):
        if not isinstance(value, string_types):
            self.fail('must_be_string', input=value, name=value)

        try:
            enum = self.enum
            if value in self.enum.enums:
                return value
            else:
                raise AttributeError()
            # return getattr(enum, value)
        except AttributeError:
            self.fail('by_name', input=value, name=value)

    # @pysnooper.snoop()
    def fail(self, key, **kwargs):
        kwargs['values'] = ', '.join(self.enum.enums)
        kwargs['names'] = ', '.join(self.enum.enums)

        if self.error:
            if self.by_value:
                kwargs['choices'] = kwargs['values']
            else:
                kwargs['choices'] = kwargs['names']
            msg = self.error.format(**kwargs)
            raise ValidationError(msg)
        else:
            super(EnumField, self).fail(key, **kwargs)
