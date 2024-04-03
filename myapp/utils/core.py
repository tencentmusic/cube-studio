"""Utility functions used across Myapp"""
from datetime import date, datetime, time, timedelta
import decimal
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
import errno
import functools
import json
import logging
import os
import signal
import copy
import sys
from time import struct_time
import traceback
from typing import List, Optional, Tuple
from urllib.parse import unquote_plus
import uuid
import zlib
import bleach
import celery
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from flask import current_app, flash, Flask, g, Markup, render_template
from flask_appbuilder.security.sqla.models import User
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_caching import Cache
import markdown as md
import numpy
import pandas as pd
import parsedatetime
from jinja2 import Environment, BaseLoader, DebugUndefined

try:
    from pydruid.utils.having import Having
except ImportError:
    pass
import sqlalchemy as sa
from sqlalchemy import event, exc, select, Text
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.sql.type_api import Variant
from sqlalchemy.types import TEXT, TypeDecorator
import math
from myapp.exceptions import MyappException, MyappTimeoutException
from myapp.utils.dates import datetime_to_epoch, EPOCH
import re

PY3K = sys.version_info >= (3, 0)
DTTM_ALIAS = "__timestamp"
ADHOC_METRIC_EXPRESSION_TYPES = {"SIMPLE": "SIMPLE", "SQL": "SQL"}

JS_MAX_INTEGER = 9007199254740991  # Largest int Java Script can handle 2^53-1

sources = {"chart": 0, "dashboard": 1, "sql_lab": 2}

try:
    # Having might not have been imported.
    class DimSelector(Having):
        def __init__(self, **args):
            # Just a hack to prevent any exceptions
            Having.__init__(self, type="equalTo", aggregation=None, value=None)

            self.having = {
                "having": {
                    "type": "dimSelector",
                    "dimension": args["dimension"],
                    "value": args["value"],
                }
            }


except NameError:
    pass


def validate_str(obj, key='var'):
    if obj and re.match("^[A-Za-z0-9_-]*$", obj):
        return True
    raise MyappException("%s is not valid" % key)


def flasher(msg, severity=None):
    """Flask's flash if available, logging call if not"""
    try:
        flash(msg, severity)
    except RuntimeError:
        if severity == "danger":
            logging.error(msg)
        else:
            logging.info(msg)


class _memoized:  # noqa
    """Decorator that caches a function's return value each time it is called

    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.

    Define ``watch`` as a tuple of attribute names if this Decorator
    should account for instance variable changes.
    """

    def __init__(self, func, watch=()):
        self.func = func
        self.cache = {}
        self.is_method = False
        self.watch = watch

    def __call__(self, *args, **kwargs):
        key = [args, frozenset(kwargs.items())]
        if self.is_method:
            key.append(tuple([getattr(args[0], v, None) for v in self.watch]))
        key = tuple(key)
        if key in self.cache:
            return self.cache[key]
        try:
            value = self.func(*args, **kwargs)
            self.cache[key] = value
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args, **kwargs)

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        if not self.is_method:
            self.is_method = True
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


def memoized(func=None, watch=None):
    if func:
        return _memoized(func)
    else:

        def wrapper(f):
            return _memoized(f, watch)

        return wrapper


def parse_js_uri_path_item(
        item: Optional[str], unquote: bool = True, eval_undefined: bool = False
) -> Optional[str]:
    """Parse a uri path item made with js.

    :param item: a uri path component
    :param unquote: Perform unquoting of string using urllib.parse.unquote_plus()
    :param eval_undefined: When set to True and item is either 'null'  or 'undefined',
    assume item is undefined and return None.
    :return: Either None, the original item or unquoted item
    """
    item = None if eval_undefined and item in ("null", "undefined") else item
    return unquote_plus(item) if unquote and item else item


def string_to_num(s: str):
    """Converts a string to an int/float

    Returns ``None`` if it can't be converted

    >>> string_to_num('5')
    5
    >>> string_to_num('5.2')
    5.2
    >>> string_to_num(10)
    10
    >>> string_to_num(10.1)
    10.1
    >>> string_to_num('this is not a string') is None
    True
    """
    if isinstance(s, (int, float)):
        return s
    if s.isdigit():
        return int(s)
    try:
        return float(s)
    except ValueError:
        return None


def list_minus(l: List, minus: List) -> List:
    """Returns l without what is in minus

    >>> list_minus([1, 2, 3], [2])
    [1, 3]
    """
    return [o for o in l if o not in minus]


def parse_human_datetime(s):
    """
    Returns ``datetime.datetime`` from human readable strings

    >>> from datetime import date, timedelta
    >>> from dateutil.relativedelta import relativedelta
    >>> parse_human_datetime('2015-04-03')
    datetime.datetime(2015, 4, 3, 0, 0)
    >>> parse_human_datetime('2/3/1969')
    datetime.datetime(1969, 2, 3, 0, 0)
    >>> parse_human_datetime('now') <= datetime.now()
    True
    >>> parse_human_datetime('yesterday') <= datetime.now()
    True
    >>> date.today() - timedelta(1) == parse_human_datetime('yesterday').date()
    True
    >>> year_ago_1 = parse_human_datetime('one year ago').date()
    >>> year_ago_2 = (datetime.now() - relativedelta(years=1) ).date()
    >>> year_ago_1 == year_ago_2
    True
    """
    if not s:
        return None
    try:
        dttm = parse(s)
    except Exception:
        try:
            cal = parsedatetime.Calendar()
            parsed_dttm, parsed_flags = cal.parseDT(s)
            # when time is not extracted, we 'reset to midnight'
            if parsed_flags & 2 == 0:
                parsed_dttm = parsed_dttm.replace(hour=0, minute=0, second=0)
            dttm = dttm_from_timetuple(parsed_dttm.utctimetuple())
        except Exception as e:
            logging.exception(e)
            raise ValueError("Couldn't parse date string [{}]".format(s))
    return dttm


def dttm_from_timetuple(d: struct_time) -> datetime:
    return datetime(d.tm_year, d.tm_mon, d.tm_mday, d.tm_hour, d.tm_min, d.tm_sec)


def parse_human_timedelta(s: str) -> timedelta:
    """
    Returns ``datetime.datetime`` from natural language time deltas

    >>> parse_human_datetime('now') <= datetime.now()
    True
    """
    cal = parsedatetime.Calendar()
    dttm = dttm_from_timetuple(datetime.now().timetuple())
    d = cal.parse(s or "", dttm)[0]
    d = datetime(d.tm_year, d.tm_mon, d.tm_mday, d.tm_hour, d.tm_min, d.tm_sec)
    return d - dttm


def parse_past_timedelta(delta_str: str) -> timedelta:
    """
    Takes a delta like '1 year' and finds the timedelta for that period in
    the past, then represents that past timedelta in positive terms.

    parse_human_timedelta('1 year') find the timedelta 1 year in the future.
    parse_past_timedelta('1 year') returns -datetime.timedelta(-365)
    or datetime.timedelta(365).
    """
    return -parse_human_timedelta(
        delta_str if delta_str.startswith("-") else f"-{delta_str}"
    )


class JSONEncodedDict(TypeDecorator):
    """Represents an immutable structure as a json-encoded string."""

    impl = TEXT

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


def datetime_f(dttm):
    """Formats datetime to take less room when it is recent"""
    if dttm:
        dttm = dttm.isoformat()
        now_iso = datetime.now().isoformat()
        if now_iso[:10] == dttm[:10]:
            dttm = dttm[11:]
        elif now_iso[:4] == dttm[:4]:
            dttm = dttm[5:]
    return "<nobr>{}</nobr>".format(dttm)


def base_json_conv(obj):
    if isinstance(obj, memoryview):
        obj = obj.tobytes()
    if isinstance(obj, numpy.int64):
        return int(obj)
    elif isinstance(obj, numpy.bool_):
        return bool(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, timedelta):
        return str(obj)
    elif isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return "[bytes]"


def json_iso_dttm_ser(obj, pessimistic: Optional[bool] = False):
    """
    json serializer that deals with dates

    >>> dttm = datetime(1970, 1, 1)
    >>> json.dumps({'dttm': dttm}, default=json_iso_dttm_ser)
    '{"dttm": "1970-01-01T00:00:00"}'
    """
    val = base_json_conv(obj)
    if val is not None:
        return val
    if isinstance(obj, (datetime, date, time, pd.Timestamp)):
        obj = obj.isoformat()
    else:
        if pessimistic:
            return "Unserializable [{}]".format(type(obj))
        else:
            raise TypeError(
                "Unserializable object {} of type {}".format(obj, type(obj))
            )
    return obj


def pessimistic_json_iso_dttm_ser(obj):
    """Proxy to call json_iso_dttm_ser in a pessimistic way

    If one of object is not serializable to json, it will still succeed"""
    return json_iso_dttm_ser(obj, pessimistic=True)


def json_int_dttm_ser(obj):
    """json serializer that deals with dates"""
    val = base_json_conv(obj)
    if val is not None:
        return val
    if isinstance(obj, (datetime, pd.Timestamp)):
        obj = datetime_to_epoch(obj)
    elif isinstance(obj, date):
        obj = (obj - EPOCH.date()).total_seconds() * 1000
    else:
        raise TypeError("Unserializable object {} of type {}".format(obj, type(obj)))
    return obj


def json_dumps_w_dates(payload):
    return json.dumps(payload, default=json_int_dttm_ser)


def error_msg_from_exception(e):
    """Translate exception into error message

    Database have different ways to handle exception. This function attempts
    to make sense of the exception object and construct a human readable
    sentence.

    TODO(bkyryliuk): parse the Presto error message from the connection
                     created via create_engine.
    engine = create_engine('presto://localhost:3506/silver') -
      gives an e.message as the str(dict)
    presto.connect('localhost', port=3506, catalog='silver') - as a dict.
    The latter version is parsed correctly by this function.
    """
    msg = ""
    if hasattr(e, "message"):
        if isinstance(e.message, dict):
            msg = e.message.get("message")
        elif e.message:
            msg = "{}".format(e.message)
    return msg or "{}".format(e)


def markdown(s: str, markup_wrap: Optional[bool] = False) -> str:
    safe_markdown_tags = [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "b",
        "i",
        "strong",
        "em",
        "tt",
        "p",
        "br",
        "span",
        "div",
        "blockquote",
        "code",
        "hr",
        "ul",
        "ol",
        "li",
        "dd",
        "dt",
        "img",
        "a",
    ]
    safe_markdown_attrs = {
        "img": ["src", "alt", "title"],
        "a": ["href", "alt", "title"],
    }
    s = md.markdown(
        s or "",
        extensions=[
            "markdown.extensions.tables",
            "markdown.extensions.fenced_code",
            "markdown.extensions.codehilite",
        ],
    )
    s = bleach.clean(s, safe_markdown_tags, safe_markdown_attrs)
    if markup_wrap:
        s = Markup(s)
    return s


def readfile(file_path: str) -> Optional[str]:
    with open(file_path) as f:
        content = f.read()
    return content


def generic_find_constraint_name(table, columns, referenced, db):
    """Utility to find a constraint name in alembic migrations"""
    t = sa.Table(table, db.metadata, autoload=True, autoload_with=db.engine)

    for fk in t.foreign_key_constraints:
        if fk.referred_table.name == referenced and set(fk.column_keys) == columns:
            return fk.name


def generic_find_fk_constraint_name(table, columns, referenced, insp):
    """Utility to find a foreign-key constraint name in alembic migrations"""
    for fk in insp.get_foreign_keys(table):
        if (
                fk["referred_table"] == referenced
                and set(fk["referred_columns"]) == columns
        ):
            return fk["name"]


def generic_find_fk_constraint_names(table, columns, referenced, insp):
    """Utility to find foreign-key constraint names in alembic migrations"""
    names = set()

    for fk in insp.get_foreign_keys(table):
        if (
                fk["referred_table"] == referenced
                and set(fk["referred_columns"]) == columns
        ):
            names.add(fk["name"])

    return names


def generic_find_uq_constraint_name(table, columns, insp):
    """Utility to find a unique constraint name in alembic migrations"""

    for uq in insp.get_unique_constraints(table):
        if columns == set(uq["column_names"]):
            return uq["name"]


def get_datasource_full_name(database_name, datasource_name, schema=None):
    if not schema:
        return "[{}].[{}]".format(database_name, datasource_name)
    return "[{}].[{}].[{}]".format(database_name, schema, datasource_name)


def dag_json_demo():
    return '''{\n   "task1_name": {},\n   "task2_name": {\n      "upstream": ["task1_name"]\n   },\n   "task3_name": {\n      "upstream": ["task2_name"]\n   }\n}'''


# job模板参数的定义
def job_template_args_definition():
    demo = f'''
{{
    "group1":{{               # {_('参数分组，仅做web显示使用')}
       "--attr1":{{           # {_('参数名')}
        "type":"str",        # str,text,list,json
        "label":"{_('参数1')}",      # {_('中文名')}
        "require":1,         # {_('是否必须')}
        "choice":[],         # {_('设定输入的可选值，可选值')}
        "default":"",        # {_('默认值')}
        "placeholder":"",    # {_('输入提示内容')}
        "describe":"{_('这里是这个参数的描述和备注')}",
        "editable":1        # {_('是否可修改')}
      }},
      "--attr2":{{
       ...
      }}
    }},
    "group2":{{
    }}
}}
'''.strip()
    return _(demo)


# 超参搜索的定义demo
def hp_parameters_demo():
    demo = '''
{                            # 标准json格式
    "--lr": {
        "type": "double",    # type，支持double、int、categorical 三种
        "min": 0.01,         # double、int类型必填min和max。max必须大于min
        "max": 0.03,
        "step": 0.01         # grid 搜索算法时，需要提供采样步长，random时不需要
    },
    "--num-layers": {
        "type": "int",
        "min": 2,
        "max": 5
    },
    "--optimizer": {
        "type": "categorical",   # categorical 类型必填list
        "list": [
            "sgd",
            "adam",
            "ftrl"
        ]
    }
}
'''.strip()

    return _(demo)


# 超参搜索的定义demo
def nni_parameters_demo():
    demo = '''
{
    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
    "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
    "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
    "momentum":{"_type":"uniform","_value":[0, 1]}
}
'''
    return demo.strip()


def validate_json(obj):
    if obj:
        try:
            json.loads(obj)
        except Exception as e:
            raise MyappException("JSON is not valid :%s" % str(e))


# 校验task args是否合法
default_args = {
    "type": "str",  # int,str,text,enum,float,multiple,dict,list
    "item_type": "str",  # 在type为enum,list时每个子属性的类型
    "label": "",  # 中文名
    "require": 1,  # 是否必须
    "choice": [],  # type为enum/multiple时，可选值
    "range": "",  # 最小最大取值，在int,float时使用  [$min,$max)
    "default": "",  # 默认值
    "placeholder": "",  # 输入提示内容
    "describe": "",
    "editable": 1  # 是否可修改
}


# 渲染字符串模板变量
def template_command(command):
    rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(command)
    des_str = rtemplate.render(rtx=g.user.username)
    return des_str


# @pysnooper.snoop()
def validate_job_args(job_template):
    validate_json(job_template.args)
    validate_job_args_type = ['int', 'bool', 'str', 'text', 'enum', 'float', 'multiple', 'dict', 'list', 'file', 'json']

    # 校验x修复参数
    # @pysnooper.snoop()
    def check_attr(attr):
        default_args_temp = copy.deepcopy(default_args)
        default_args_temp.update(attr)
        attr = default_args_temp
        attr['type'] = str(attr['type'])
        attr['item_type'] = str(attr['item_type'])
        attr['label'] = str(attr['label'])
        attr['require'] = int(attr['require'])
        attr['range'] = str(attr['range'])
        attr['default'] = attr['default']
        attr['placeholder'] = str(attr['placeholder'])
        attr['describe'] = str(attr['describe']) if attr['describe'] else str(attr['label'])
        attr['editable'] = int(attr['editable'])


        if attr['type'] not in validate_job_args_type:
            raise MyappException("job template args type must in %s " % str(validate_job_args_type))
        if attr['type'] == 'enum' or attr['type'] == 'multiple' or attr['type'] == 'list':
            if attr['item_type'] not in ['int', 'str', 'text', 'float', 'dict']:
                raise MyappException("job template args item_type must in %s " % str(['int', 'str', 'text', 'float', 'dict']))
        if attr['type'] == 'enum' or attr['type'] == 'multiple':
            if not attr['choice']:
                raise MyappException("job template args choice must exist when type is enum,multiple ")
        if attr['type'] == 'dict':
            if not attr['sub_args']:
                raise MyappException("job template args sub_args must exist when type is dict ")
            for sub_attr in attr['sub_args']:
                attr['sub_args'][sub_attr] = check_attr(attr['sub_args'][sub_attr])
        if attr['type'] == 'list' and attr['item_type'] == 'dict':
            if not attr['sub_args']:
                raise MyappException("job template args sub_args must exist when type is list ")

            for sub_attr in attr['sub_args']:
                attr['sub_args'][sub_attr] = check_attr(attr['sub_args'][sub_attr])

        return attr

    args = json.dumps(json.loads(job_template.args), indent=4, ensure_ascii=False)
    job_args = json.loads(args)

    for group in job_args:
        for attr_name in job_args[group]:
            job_args[group][attr_name] = check_attr(job_args[group][attr_name])

    return json.dumps(job_args, indent=4, ensure_ascii=False)


# task_args 为用户填写的参数，job_args为定义的参数标准
# @pysnooper.snoop()
def validate_task_args(task_args, job_args):  # 两个都是字典
    if not task_args:
        return {}

    # @pysnooper.snoop()
    def to_value(value, value_type):
        if value_type == 'str' or value_type == 'text':
            return str(value)
        if value_type == 'int':
            return int(value)
        if value_type == 'float':
            return float(value)
        if value_type == 'dict':
            return dict(value)
        if value_type == 'json':
            return value
            # try:
            #     return json.dumps(json.loads(value),indent=4,ensure_ascii=False)
            # except Exception as e:
            #     raise MyappException("task args json is not valid: %s"%str(value))

        raise MyappException("task args type is not valid: %s" % str(value))

    # 校验 task的attr和job的attr是否符合
    # @pysnooper.snoop()
    def check_attr(task_attr, job_attr):
        validate_attr = task_attr

        if job_attr['type'] == 'str' or job_attr['type'] == 'text' or job_attr['type'] == 'int' or job_attr['type'] == 'float':
            validate_attr = to_value(task_attr, job_attr['type'])

        if job_attr['type'] == 'json':
            validate_attr = to_value(task_attr, job_attr['type'])

        if job_attr['type'] == 'enum':
            validate_attr = to_value(task_attr, job_attr['item_type'])
            if validate_attr not in job_attr['choice']:
                raise MyappException("task arg type(enum) is not in choice: %s" % job_attr['choice'])

        if job_attr['type'] == 'multiple':
            if type(task_attr) == str:
                validate_attr = re.split(' |,|;|\n|\t', str(task_attr))  # 分割字符串
            if type(validate_attr) == list:
                for item in validate_attr:
                    if item not in job_attr['choice']:
                        raise MyappException("task arg type(enum) is not in choice: %s" % job_attr['choice'])
        if job_attr['type'] == 'dict' and type(job_attr['sub_args']) != dict:
            raise MyappException("task args type(dict) sub args must is dict")

        # 校验字典的子属性
        if job_attr['type'] == 'dict':
            for sub_attr_name in task_attr:
                validate_attr[sub_attr_name] = check_attr(task_attr[sub_attr_name], job_attr['sub_args'][sub_attr_name])

        # 检验list的每个元素
        if job_attr['type'] == 'list':
            if job_attr['item_type'] == 'dict':
                validate_attr = []
                for sub_task_attr in task_attr:
                    validate_sub_attr = {}
                    for sub_attr_name in sub_task_attr:
                        validate_sub_attr[sub_attr_name] = check_attr(sub_task_attr[sub_attr_name],job_attr['sub_args'][sub_attr_name])
                    validate_attr.append(validate_sub_attr)
            else:
                validate_attr = [to_value(sub_task_attr, job_attr['item_type']) for sub_task_attr in task_attr]

        return validate_attr

    validate_args = {}
    try:
        for group in job_args:
            for attr_name in job_args[group]:
                job_attr = job_args[group][attr_name]
                if attr_name in task_args:
                    task_attr = task_args[attr_name]
                    validate_args[attr_name] = check_attr(task_attr, job_attr)
                elif job_args['require']:
                    raise MyappException("task args %s must is require" % attr_name)
                elif job_args['default']:
                    validate_args[attr_name] = job_args['default']

        return validate_args


    except Exception as e:
        raise MyappException("task args is not valid: %s" % str(e))


def up_word(words):
    words = re.split('_| |,|;|\n|\t', str(words))  # 分割字符串
    return ' '.join([s.capitalize() for s in words])


def add_column():
    pass


def table_has_constraint(table, name, db):
    """Utility to find a constraint name in alembic migrations"""
    t = sa.Table(table, db.metadata, autoload=True, autoload_with=db.engine)

    for c in t.constraints:
        if c.name == name:
            return True
    return False


class timeout:
    """
    To be used in a ``with`` block and timeout its content.
    """

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        logging.error("Process timed out")
        raise MyappTimeoutException(self.error_message)

    def __enter__(self):
        try:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        except ValueError as e:
            logging.warning("timeout can't be used in the current context")
            logging.exception(e)

    def __exit__(self, type, value, traceback):
        try:
            signal.alarm(0)
        except ValueError as e:
            logging.warning("timeout can't be used in the current context")
            logging.exception(e)


def pessimistic_connection_handling(some_engine):
    @event.listens_for(some_engine, "engine_connect")
    def ping_connection(connection, branch):
        if branch:
            # 'branch' refers to a sub-connection of a connection,
            # we don't want to bother pinging on these.
            return

        # turn off 'close with result'.  This flag is only used with
        # 'connectionless' execution, otherwise will be False in any case
        save_should_close_with_result = connection.should_close_with_result
        connection.should_close_with_result = False

        try:
            # run a SELECT 1.   use a core select() so that
            # the SELECT of a scalar value without a table is
            # appropriately formatted for the backend
            connection.scalar(select([1]))
        except exc.DBAPIError as err:
            # catch SQLAlchemy's DBAPIError, which is a wrapper
            # for the DBAPI's exception.  It includes a .connection_invalidated
            # attribute which specifies if this connection is a 'disconnect'
            # condition, which is based on inspection of the original exception
            # by the dialect in use.
            if err.connection_invalidated:
                # run the same SELECT again - the connection will re-validate
                # itself and establish a new connection.  The disconnect detection
                # here also causes the whole connection pool to be invalidated
                # so that all stale connections are discarded.
                connection.scalar(select([1]))
            else:
                raise
        finally:
            # restore 'close with result'
            connection.should_close_with_result = save_should_close_with_result


class QueryStatus:
    """Enum-type class for query statuses"""

    STOPPED = "stopped"
    FAILED = "failed"
    PENDING = "pending"
    RUNNING = "running"
    SCHEDULED = "scheduled"
    SUCCESS = "success"
    TIMED_OUT = "timed_out"


def notify_user_about_perm_udate(granter, user, role, datasource, tpl_name, config):
    msg = render_template(
        tpl_name, granter=granter, user=user, role=role, datasource=datasource
    )
    logging.info(msg)
    subject = __(
        "[Myapp] Access to the datasource %(name)s was granted",
        name=datasource.full_name,
    )
    send_email_smtp(
        user.email,
        subject,
        msg,
        config,
        bcc=granter.email,
        dryrun=not config.get("EMAIL_NOTIFICATIONS"),
    )


def send_email_smtp(
        to,
        subject,
        html_content,
        config,
        files=None,
        data=None,
        images=None,
        dryrun=False,
        cc=None,
        bcc=None,
        mime_subtype="mixed",
):
    """
    Send an email with html content, eg:
    send_email_smtp(
        'test@example.com', 'foo', '<b>Foo</b> bar',['/dev/null'], dryrun=True)
    """
    smtp_mail_from = config.get("SMTP_MAIL_FROM")
    to = get_email_address_list(to)

    msg = MIMEMultipart(mime_subtype)
    msg["Subject"] = subject
    msg["From"] = smtp_mail_from
    msg["To"] = ", ".join(to)
    msg.preamble = "This is a multi-part message in MIME format."

    recipients = to
    if cc:
        cc = get_email_address_list(cc)
        msg["CC"] = ", ".join(cc)
        recipients = recipients + cc

    if bcc:
        # don't add bcc in header
        bcc = get_email_address_list(bcc)
        recipients = recipients + bcc

    msg["Date"] = formatdate(localtime=True)
    mime_text = MIMEText(html_content, "html")
    msg.attach(mime_text)

    # Attach files by reading them from disk
    for fname in files or []:
        basename = os.path.basename(fname)
        with open(fname, "rb") as f:
            msg.attach(
                MIMEApplication(
                    f.read(),
                    Content_Disposition="attachment; filename='%s'" % basename,
                    Name=basename,
                )
            )

    # Attach any files passed directly
    for name, body in (data or {}).items():
        msg.attach(
            MIMEApplication(
                body, Content_Disposition="attachment; filename='%s'" % name, Name=name
            )
        )

    # Attach any inline images, which may be required for display in
    # HTML content (inline)
    for msgid, body in (images or {}).items():
        image = MIMEImage(body)
        image.add_header("Content-ID", "<%s>" % msgid)
        image.add_header("Content-Disposition", "inline")
        msg.attach(image)

    send_MIME_email(smtp_mail_from, recipients, msg, config, dryrun=dryrun)


def send_MIME_email(e_from, e_to, mime_msg, config, dryrun=False):
    logging.info("Dryrun enabled, email notification content is below:")
    logging.info(mime_msg.as_string())


# 自动将,;\n分割符变为列表
def get_email_address_list(address_string: str) -> List[str]:
    address_string_list: List[str] = []
    if isinstance(address_string, str):
        if "," in address_string:
            address_string_list = address_string.split(",")
        elif "\n" in address_string:
            address_string_list = address_string.split("\n")
        elif ";" in address_string:
            address_string_list = address_string.split(";")
        else:
            address_string_list = [address_string]
    return [x.strip() for x in address_string_list if x.strip()]


def choicify(values):
    """Takes an iterable and makes an iterable of tuples with it"""
    return [(v, v) for v in values]


def setup_cache(app: Flask, cache_config) -> Optional[Cache]:
    """Setup the flask-cache on a flask app"""
    if cache_config:
        if isinstance(cache_config, dict):
            if cache_config.get("CACHE_TYPE") != "null":
                return Cache(app, config=cache_config)
        else:
            # Accepts a custom cache initialization function,
            # returning an object compatible with Flask-Caching API
            return cache_config(app)

    return None


def zlib_compress(data):
    """
    Compress things in a py2/3 safe fashion
    >>> json_str = '{"test": 1}'
    >>> blob = zlib_compress(json_str)
    """
    if PY3K:
        if isinstance(data, str):
            return zlib.compress(bytes(data, "utf-8"))
        return zlib.compress(data)
    return zlib.compress(data)


def zlib_decompress_to_string(blob):
    """
    Decompress things to a string in a py2/3 safe fashion
    >>> json_str = '{"test": 1}'
    >>> blob = zlib_compress(json_str)
    >>> got_str = zlib_decompress_to_string(blob)
    >>> got_str == json_str
    True
    """
    if PY3K:
        if isinstance(blob, bytes):
            decompressed = zlib.decompress(blob)
        else:
            decompressed = zlib.decompress(bytes(blob, "utf-8"))
        return decompressed.decode("utf-8")
    return zlib.decompress(blob)


_celery_app = None


# 从CELERY_CONFIG中获取定义celery app 的配置（全局设置每个任务的配置，所有对每个worker配置都是多的，因为不同worker只处理部分task）
def get_celery_app(config):
    global _celery_app
    if _celery_app:
        return _celery_app
    _celery_app = celery.Celery()
    _celery_app.config_from_object(config.get("CELERY_CONFIG"))
    _celery_app.set_default()
    return _celery_app


def to_adhoc(filt, expressionType="SIMPLE", clause="where"):
    result = {
        "clause": clause.upper(),
        "expressionType": expressionType,
        "filterOptionName": str(uuid.uuid4()),
    }

    if expressionType == "SIMPLE":
        result.update(
            {
                "comparator": filt.get("val"),
                "operator": filt.get("op"),
                "subject": filt.get("col"),
            }
        )
    elif expressionType == "SQL":
        result.update({"sqlExpression": filt.get(clause)})

    return result


def merge_extra_filters(form_data: dict):
    # extra_filters are temporary/contextual filters (using the legacy constructs)
    # that are external to the slice definition. We use those for dynamic
    # interactive filters like the ones emitted by the "Filter Box" visualization.
    # Note extra_filters only support simple filters.
    if "extra_filters" in form_data:
        # __form and __to are special extra_filters that target time
        # boundaries. The rest of extra_filters are simple
        # [column_name in list_of_values]. `__` prefix is there to avoid
        # potential conflicts with column that would be named `from` or `to`
        if "adhoc_filters" not in form_data or not isinstance(form_data["adhoc_filters"], list
        ):
            form_data["adhoc_filters"] = []
        date_options = {
            "__time_range": "time_range",
            "__time_col": "granularity_sqla",
            "__time_grain": "time_grain_sqla",
            "__time_origin": "druid_time_origin",
            "__granularity": "granularity",
        }

        # Grab list of existing filters 'keyed' on the column and operator

        def get_filter_key(f):
            if "expressionType" in f:
                return "{}__{}".format(f["subject"], f["operator"])
            else:
                return "{}__{}".format(f["col"], f["op"])

        existing_filters = {}
        for existing in form_data["adhoc_filters"]:
            if (
                    existing["expressionType"] == "SIMPLE"
                    and existing["comparator"] is not None
                    and existing["subject"] is not None
            ):
                existing_filters[get_filter_key(existing)] = existing["comparator"]

        for filtr in form_data["extra_filters"]:
            # Pull out time filters/options and merge into form data
            if date_options.get(filtr["col"]):
                if filtr.get("val"):
                    form_data[date_options[filtr["col"]]] = filtr["val"]
            elif filtr["val"]:
                # Merge column filters
                filter_key = get_filter_key(filtr)
                if filter_key in existing_filters:
                    # Check if the filter already exists
                    if isinstance(filtr["val"], list):
                        if isinstance(existing_filters[filter_key], list):
                            # Add filters for unequal lists
                            # order doesn't matter
                            if sorted(existing_filters[filter_key]) != sorted(
                                    filtr["val"]
                            ):
                                form_data["adhoc_filters"].append(to_adhoc(filtr))
                        else:
                            form_data["adhoc_filters"].append(to_adhoc(filtr))
                    else:
                        # Do not add filter if same value already exists
                        if filtr["val"] != existing_filters[filter_key]:
                            form_data["adhoc_filters"].append(to_adhoc(filtr))
                else:
                    # Filter not found, add it
                    form_data["adhoc_filters"].append(to_adhoc(filtr))
        # Remove extra filters from the form data since no longer needed
        del form_data["extra_filters"]


def merge_request_params(form_data: dict, params: dict):
    url_params = {}
    for key, value in params.items():
        if key in ("form_data", "r"):
            continue
        url_params[key] = value
    form_data["url_params"] = url_params


def user_label(user: User) -> Optional[str]:
    """Given a user ORM FAB object, returns a label"""
    return user.username

    # if user:
    #     if user.first_name and user.last_name:
    #         return user.first_name + " " + user.last_name
    #     else:
    #         return user.username
    # return None


def is_adhoc_metric(metric) -> bool:
    return (
            isinstance(metric, dict)
            and (
                    (
                            metric["expressionType"] == ADHOC_METRIC_EXPRESSION_TYPES["SIMPLE"]
                            and metric["column"]
                            and metric["aggregate"]
                    )
                    or (
                            metric["expressionType"] == ADHOC_METRIC_EXPRESSION_TYPES["SQL"]
                            and metric["sqlExpression"]
                    )
            )
            and metric["label"]
    )


def ensure_path_exists(path: str):
    try:
        os.makedirs(path)
    except OSError as exc:
        if not (os.path.isdir(path) and exc.errno == errno.EEXIST):
            raise


def get_since_until(
        time_range: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        time_shift: Optional[str] = None,
        relative_start: Optional[str] = None,
        relative_end: Optional[str] = None,
) -> Tuple[datetime, datetime]:
    """Return `since` and `until` date time tuple from string representations of
    time_range, since, until and time_shift.

    This functiom supports both reading the keys separately (from `since` and
    `until`), as well as the new `time_range` key. Valid formats are:

        - ISO 8601
        - X days/years/hours/day/year/weeks
        - X days/years/hours/day/year/weeks ago
        - X days/years/hours/day/year/weeks from now
        - freeform

    Additionally, for `time_range` (these specify both `since` and `until`):

        - Last day
        - Last week
        - Last month
        - Last quarter
        - Last year
        - No filter
        - Last X seconds/minutes/hours/days/weeks/months/years
        - Next X seconds/minutes/hours/days/weeks/months/years

    """
    separator = " : "
    relative_start = parse_human_datetime(relative_start if relative_start else "today")
    relative_end = parse_human_datetime(relative_end if relative_end else "today")
    common_time_frames = {
        "Last day": (
            relative_start - relativedelta(days=1),  # noqa: T400
            relative_end,
        ),
        "Last week": (
            relative_start - relativedelta(weeks=1),  # noqa: T400
            relative_end,
        ),
        "Last month": (
            relative_start - relativedelta(months=1),  # noqa: T400
            relative_end,
        ),
        "Last quarter": (
            relative_start - relativedelta(months=3),  # noqa: T400
            relative_end,
        ),
        "Last year": (
            relative_start - relativedelta(years=1),  # noqa: T400
            relative_end,
        ),
    }

    if time_range:
        if separator in time_range:
            since, until = time_range.split(separator, 1)
            if since and since not in common_time_frames:
                since = add_ago_to_since(since)
            since = parse_human_datetime(since)
            until = parse_human_datetime(until)
        elif time_range in common_time_frames:
            since, until = common_time_frames[time_range]
        elif time_range == "No filter":
            since = until = None
        else:
            rel, num, grain = time_range.split()
            if rel == "Last":
                since = relative_start - relativedelta(  # noqa: T400
                    **{grain: int(num)}
                )
                until = relative_end
            else:  # rel == 'Next'
                since = relative_start
                until = relative_end + relativedelta(**{grain: int(num)})  # noqa: T400
    else:
        since = since or ""
        if since:
            since = add_ago_to_since(since)
        since = parse_human_datetime(since)
        until = parse_human_datetime(until) if until else relative_end

    if time_shift:
        time_delta = parse_past_timedelta(time_shift)
        since = since if since is None else (since - time_delta)  # noqa: T400
        until = until if until is None else (until - time_delta)  # noqa: T400

    if since and until and since > until:
        raise ValueError(_("From date cannot be larger than to date"))

    return since, until  # noqa: T400


def add_ago_to_since(since: str) -> str:
    """
    Backwards compatibility hack. Without this slices with since: 7 days will
    be treated as 7 days in the future.

    :param str since:
    :returns: Since with ago added if necessary
    :rtype: str
    """
    since_words = since.split(" ")
    grains = ["days", "years", "hours", "day", "year", "weeks"]
    if len(since_words) == 2 and since_words[1] in grains:
        since += " ago"
    return since


def convert_legacy_filters_into_adhoc(fd):
    mapping = {"having": "having_filters", "where": "filters"}

    if not fd.get("adhoc_filters"):
        fd["adhoc_filters"] = []

        for clause, filters in mapping.items():
            if clause in fd and fd[clause] != "":
                fd["adhoc_filters"].append(to_adhoc(fd, "SQL", clause))

            if filters in fd:
                for filt in filter(lambda x: x is not None, fd[filters]):
                    fd["adhoc_filters"].append(to_adhoc(filt, "SIMPLE", clause))

    for key in ("filters", "having", "having_filters", "where"):
        if key in fd:
            del fd[key]


def split_adhoc_filters_into_base_filters(fd):
    """
    Mutates form data to restructure the adhoc filters in the form of the four base
    filters, `where`, `having`, `filters`, and `having_filters` which represent
    free form where sql, free form having sql, structured where clauses and structured
    having clauses.
    """
    adhoc_filters = fd.get("adhoc_filters")
    if isinstance(adhoc_filters, list):
        simple_where_filters = []
        simple_having_filters = []
        sql_where_filters = []
        sql_having_filters = []
        for adhoc_filter in adhoc_filters:
            expression_type = adhoc_filter.get("expressionType")
            clause = adhoc_filter.get("clause")
            if expression_type == "SIMPLE":
                if clause == "WHERE":
                    simple_where_filters.append(
                        {
                            "col": adhoc_filter.get("subject"),
                            "op": adhoc_filter.get("operator"),
                            "val": adhoc_filter.get("comparator"),
                        }
                    )
                elif clause == "HAVING":
                    simple_having_filters.append(
                        {
                            "col": adhoc_filter.get("subject"),
                            "op": adhoc_filter.get("operator"),
                            "val": adhoc_filter.get("comparator"),
                        }
                    )
            elif expression_type == "SQL":
                if clause == "WHERE":
                    sql_where_filters.append(adhoc_filter.get("sqlExpression"))
                elif clause == "HAVING":
                    sql_having_filters.append(adhoc_filter.get("sqlExpression"))
        fd["where"] = " AND ".join(["({})".format(sql) for sql in sql_where_filters])
        fd["having"] = " AND ".join(["({})".format(sql) for sql in sql_having_filters])
        fd["having_filters"] = simple_having_filters
        fd["filters"] = simple_where_filters


def get_username() -> Optional[str]:
    """Get username if within the flask context, otherwise return noffin'"""
    try:
        return g.user.username
    except Exception:
        return None


def MediumText() -> Variant:
    return Text().with_variant(MEDIUMTEXT(), "mysql")


def shortid() -> str:
    return "{}".format(uuid.uuid4())[-12:]


def get_stacktrace():
    if current_app.config.get("SHOW_STACKTRACE"):
        return traceback.format_exc()


def pic2html(image_arr,max_num=4):
    # 要定格写，不然markdown不识别
    html = '''
<table> 
<tbody>
%s
</tbody>
</table>
    '''
    if len(image_arr)==1:
        one_img_html = f'''
<tr> 
 {'<td> <a href="%s"><img width="300px" src="%s" /></a> </td>'%(image_arr[0],image_arr[0])}
</tr> 
        '''
        html = html%one_img_html
    if len(image_arr)==2:
        one_img_html = f'''
<tr> 
 {'<td> <a href="%s"><img width="300px" src="%s" /></a> </td>' % (image_arr[0], image_arr[0])}
 {'<td> <a href="%s"><img width="300px" src="%s" /></a> </td>' % (image_arr[1], image_arr[1])}
</tr> 
        '''
        html = html % one_img_html
    if len(image_arr)==3 and max_num<5:
        one_img_html = f'''
<tr> 
 {'<td> <a href="%s"><img width="300px" src="%s" /></a> </td>' % (image_arr[0], image_arr[0])}
 {'<td> <a href="%s"><img width="300px" src="%s" /></a> </td>' % (image_arr[1], image_arr[1])}
</tr> 
<tr> 
 {'<td> <a href="%s"><img width="300px" src="%s" /></a> </td>' % (image_arr[2], image_arr[2])}
</tr> 
        '''
        html = html % one_img_html
    if len(image_arr)==4 and max_num<5:
        one_img_html = f'''
<tr> 
 {'<td> <a href="%s"><img width="300px" src="%s" /></a> </td>' % (image_arr[0], image_arr[0])}
 {'<td> <a href="%s"><img width="300px" src="%s" /></a> </td>' % (image_arr[1], image_arr[1])}
</tr> 
<tr> 
 {'<td> <a href="%s"><img width="300px" src="%s" /></a> </td>' % (image_arr[2], image_arr[2])}
 {'<td> <a href="%s"><img width="300px" src="%s" /></a> </td>' % (image_arr[3], image_arr[3])}
</tr> 
        '''
        html = html % one_img_html
    return html

import pysnooper

# @pysnooper.snoop()
def check_resource_memory(resource_memory, src_resource_memory=None):
    from myapp import conf

    # @pysnooper.snoop()
    def check_max_memory(resource, src_resource=None):
        if not resource:
            return resource
        src_resource_int = 0
        if src_resource and 'G' in src_resource:
            src_resource_int = int(float(src_resource.replace('G', '')) * 1000)
        elif src_resource and 'M' in src_resource:
            src_resource_int = int(src_resource.replace('M', ''))
        elif src_resource:
            src_resource_int = float(src_resource)

        src_resource_int = math.ceil(src_resource_int/1000)

        resource_int = 0
        if resource and 'G' in resource:
            resource_int = int(float(resource.replace('G', '')) * 1000)
        if resource and 'M' in resource:
            resource_int = int(resource.replace('M', ''))

        resource_int = math.ceil(resource_int/1000)

        MAX_TASK_MEM = conf.get('MAX_TASK_MEM', 100)
        #  如果是变小了，可以直接使用，这是因为最大值可能是admin设置的。普通用户只能调小
        if resource_int <= src_resource_int:
            return resource

        if resource_int >= MAX_TASK_MEM >= src_resource_int:
            # flash(_('占用memory算力超过管理员设定的最大值，系统已自动配置memory为最大值') + str(MAX_TASK_MEM), category='warning')
            return f"{MAX_TASK_MEM}G"

        if resource_int >=src_resource_int >=MAX_TASK_MEM:
            # flash(_('占用memory算力超过管理员设定的最大值，系统已自动配置memory为原有值') + str(math.ceil(src_resource_int))+"G",category='warning')
            return f"{math.ceil(src_resource_int)}G"

        return resource

    resource = resource_memory.upper().replace('-', '~').replace('_', '~').strip()
    if not "~" in resource:
        pattern = '^[0-9]+[GM]$'  # 匹配字符串
        match_obj = re.match(pattern=pattern, string=resource)
        if not match_obj:
            raise MyappException('resource memory input not valid')
        if hasattr(g,'user') and not g.user.is_admin():
            resource = check_max_memory(resource, src_resource_memory)
        else:
            resource = resource
        return resource
    else:
        pattern = '^[0-9]+[GM]~[0-9]+[GM]$'  # 匹配字符串
        match_obj = re.match(pattern=pattern, string=resource)
        if not match_obj:
            raise MyappException('resource memory input not valid')
        if hasattr(g,'user') and not g.user.is_admin():
            min = check_max_memory(resource.split('~')[0])
            max = check_max_memory(resource.split('~')[1])
            resource = str(min) + "~" + str(max)
        else:
            min = resource.split('~')[0]
            max = resource.split('~')[1]
            resource = str(min) + "~" + str(max)
        return resource


def check_resource_cpu(resource_cpu, src_resource_cpu=None):
    from myapp import conf

    def check_max_cpu(resource, src_resource=None):
        resource_int = float(resource) if resource else 0
        src_resource_int = float(src_resource) if src_resource else 0
        MAX_TASK_CPU = float(conf.get('MAX_TASK_CPU',50))

        if resource_int <= src_resource_int:
            return resource

        if resource_int >= src_resource_int >= MAX_TASK_CPU:
            # flash(_('占用cpu算力超过管理员设定的最大值，系统已自动配置cpu为原有值') + str(src_resource), category='warning')
            return src_resource

        if resource_int >=MAX_TASK_CPU >=src_resource_int:
            # flash(_('占用cpu算力超过管理员设定的最大值，系统已自动配置cpu为最大值') + str(MAX_TASK_CPU),category='warning')
            return str(MAX_TASK_CPU)

        return resource

    resource = resource_cpu.upper().replace('-', '~').replace('_', '~').strip()
    if not "~" in resource:
        pattern = '^[0-9\\.]+$'  # 匹配字符串
        match_obj = re.match(pattern=pattern, string=resource)
        if not match_obj:
            raise MyappException('resource cpu input not valid')
        if hasattr(g,'user') and not g.user.is_admin():
            resource = check_max_cpu(resource, src_resource_cpu)
        else:
            resource = resource
        return resource
    else:
        pattern = '^[0-9\\.]+~[0-9\\.]+$'  # 匹配字符串
        match_obj = re.match(pattern=pattern, string=resource)
        if not match_obj:
            raise MyappException('resource cpu input not valid')
        try:
            resource = "%.1f~%.1f" % (float(resource.split("~")[0]), float(resource.split("~")[1]))
        except Exception:
            raise MyappException('resource cpu input not valid')
        if hasattr(g,'user') and not g.user.is_admin():
            min = check_max_cpu(resource.split('~')[0])
            max = check_max_cpu(resource.split('~')[1])
            resource = str(min) + "~" + str(max)
        else:
            min = resource.split('~')[0]
            max = resource.split('~')[1]
            resource = str(min) + "~" + str(max)
        return resource


def check_resource_gpu(resource_gpu, src_resource_gpu=None):
    from myapp import conf

    def check_max_gpu(gpu_num, src_gpu_num=0):

        MAX_TASK_GPU = int(conf.get('MAX_TASK_GPU',8))

        if gpu_num <= src_gpu_num:
            return gpu_num

        if gpu_num >= src_gpu_num >= MAX_TASK_GPU:
            return src_gpu_num

        if gpu_num >=MAX_TASK_GPU >=src_gpu_num:
            return str(MAX_TASK_GPU)

        return gpu_num

    gpu_num, gpu_type, resource_name = get_gpu(resource_gpu)
    src_gpu_num,_,_ = get_gpu(src_resource_gpu) if src_resource_gpu else 0,0,0

    if hasattr(g,'user')  and not g.user.is_admin():
        resource_gpu = check_max_gpu(gpu_num, src_gpu_num)
        if math.ceil(float(resource_gpu))==resource_gpu:
            resource_gpu = math.ceil(float(resource_gpu))
        if gpu_type:
            resource_gpu +=f'({gpu_type})'
    else:
        resource_gpu = resource_gpu

    return str(resource_gpu)



def check_resource(resource_memory,resource_cpu,resource_gpu,src_resource_memory=None,src_resource_cpu=None,src_resource_gpu=None):
    new_resource_memory = check_resource_memory(resource_memory,src_resource_memory)
    new_resource_memory = str(math.ceil(float(str(new_resource_memory).replace('G',''))))+"G"
    new_resource_cpu = str(math.ceil(float(check_resource_cpu(resource_cpu,src_resource_cpu))))
    new_resource_gpu = str(check_resource_gpu(resource_gpu, src_resource_gpu))

    if str(new_resource_memory).replace('G', '') != str(resource_memory).replace('G', '') or str(new_resource_cpu) != str(resource_cpu) or str(new_resource_gpu) != str(resource_gpu):
        message = _('占用算力超过管理员设定的最大值，系统已自动调整算力为:') + f"cpu({new_resource_cpu}),memory({new_resource_memory}),gpu({new_resource_gpu})"
        flash(message=message,category='warning')

    return new_resource_memory,new_resource_cpu,new_resource_gpu

def checkip(ip):
    import re
    if ":" in ip:
        ip = ip[:ip.index(':')]
    if '|' in ip:
        ip = ip.split('|')[0]   # 内网ip|公网域名
    p = re.compile('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$')
    if p.match(ip):
        return True
    else:
        return False

import pysnooper

# @pysnooper.snoop()
def get_gpu(resource_gpu,resource_name=None):
    from myapp import conf
    gpu_num = 0
    if not resource_name:
        resource_name=conf.get('DEFAULT_GPU_RESOURCE_NAME','')
    gpu_type = None
    try:
        if resource_gpu:
            if '(' in resource_gpu:
                gpu_type = re.findall(r"\((.+?)\)", resource_gpu)
                gpu_type = gpu_type[0] if gpu_type else None
            if '（' in resource_gpu:
                gpu_type = re.findall(r"（(.+?)）", resource_gpu)
                gpu_type = gpu_type[0] if gpu_type else None

            # 括号里面填的可能是npu，这种词汇，不属于卡的型号，而是卡的类型
            if gpu_type and gpu_type.lower() in list(conf.get('GPU_RESOURCE',{}).keys()):
                gpu_mfrs=gpu_type.lower()
                gpu_type=None
                resource_name = conf.get("GPU_RESOURCE", {}).get(gpu_mfrs, resource_name)
            # 填的是(卡的类型,卡的型号)
            if gpu_type and ',' in gpu_type:
                gpu_mfrs=gpu_type.split(',')[0].strip().lower()
                if gpu_mfrs:
                    resource_name = conf.get("GPU_RESOURCE", {}).get(gpu_mfrs, resource_name)
                gpu_type=gpu_type.split(',')[1].strip().upper()

            resource_gpu = resource_gpu[0:resource_gpu.index('(')] if '(' in resource_gpu else resource_gpu
            resource_gpu = resource_gpu[0:resource_gpu.index('（')] if '（' in resource_gpu else resource_gpu
            gpu_num = float(resource_gpu)

    except Exception as e:
        print(e)
    gpu_type = gpu_type.upper() if gpu_type else None
    return gpu_num, gpu_type, resource_name

# 按expand字段中index字段进行排序
def sort_expand_index(items):
    all = {
        0: []
    }
    for item in items:
        try:
            if item.expand:
                index = float(json.loads(item.expand).get('index', 0))
                if index:
                    if index in all:
                        all[index].append(item)
                    else:
                        all[index] = [item]
                else:
                    all[0].append(item)
            else:
                all[0].append(item)
        except Exception as e:
            print(e)
    back = []
    for index in sorted(all):
        back.extend(all[index])
        # 当有小数的时候自动转正
        # if float(index)!=int(index):
        #     pass
    return back


# 生成前端锁需要的扩展字段
# @pysnooper.snoop()
def fix_task_position(pipeline, tasks, expand_tasks):
    for task_name in tasks:
        task = tasks[task_name]

    # print(pipeline['dag_json'])
    dag_json = json.loads(pipeline['dag_json'])
    dag_json_sorted = sorted(dag_json.items(), key=lambda item: item[0])
    dag_json = {}
    for item in dag_json_sorted:
        dag_json[item[0]] = item[1]

    # 设置节点的位置
    def set_position(task_id, x, y):
        for task in expand_tasks:
            if str(task_id) == task['id']:
                task['position']['x'] = x
                task['position']['y'] = y

    def read_position(task_id):
        for task in expand_tasks:
            if str(task_id) == task['id']:
                return task['position']['x'], task['position']['y']

    # 检查指定位置是否存在指定节点
    def has_exist_node(x, y, task_id):
        for task in expand_tasks:
            if 'position' in task:
                if int(x) == int(task['position']['x']) and int(y) == int(task['position']['y']) and task['id'] != str(task_id):
                    return True
        return False

    # 生成下行链路图
    for task_name in dag_json:
        dag_json[task_name]['downstream'] = []
        for task_name1 in dag_json:
            if task_name in dag_json[task_name1].get("upstream", []):
                dag_json[task_name]['downstream'].append(task_name1)

    # 获取节点下游节点总数目
    def get_down_node_num(task_name):
        down_nodes = dag_json[task_name].get('downstream', [])
        if down_nodes:
            return len(down_nodes) + sum([get_down_node_num(node) for node in down_nodes])
        else:
            return 0

    # 计算每个根节点的下游叶子总数
    has_change = True
    root_num = 0
    root_nodes = []
    for task_name in dag_json:
        task = dag_json[task_name]
        # 为根节点记录第几颗树和deep
        if not task.get("upstream", []):
            root_num += 1
            task['deep'] = 1
            root_nodes.append(task_name)
            dag_json[task_name]['total_down_num'] = get_down_node_num(task_name)

    root_nodes = sorted(root_nodes, key=lambda task_name: dag_json[task_name]['total_down_num'], reverse=True)  # 按子孙数量排序
    print(root_nodes)
    for i in range(len(root_nodes)):
        dag_json[root_nodes[i]]['index'] = i

    # 更新叶子深度和树index，下游节点总数目
    max_deep = 1
    while (has_change):
        has_change = False
        for task_name in dag_json:
            task = dag_json[task_name]
            downstream_tasks = dag_json[task_name]['downstream']

            # 配置全部下游节点总数
            if 'total_down_num' not in dag_json[task_name]:
                has_change = True
                dag_json[task_name]['total_down_num'] = get_down_node_num(task_name)

            for downstream_task_name in downstream_tasks:
                # 新出现的叶子节点，直接deep+1
                if 'deep' not in dag_json[downstream_task_name]:
                    has_change = True
                    if 'deep' in task:
                        dag_json[downstream_task_name]['deep'] = 1 + task['deep']
                        if max_deep < (1 + task['deep']):
                            max_deep = 1 + task['deep']
                else:
                    # 旧叶子，可能节点被多个不同deep的上游引导，使用deep最大的做为引导
                    if dag_json[downstream_task_name]['deep'] < task['deep'] + 1:
                        has_change = True
                        dag_json[downstream_task_name]['deep'] = 1 + task['deep']
                        if max_deep < (1 + task['deep']):
                            max_deep = 1 + task['deep']

                # 叶子节点直接采用根节点的信息。有可能是多个根长出来的，选择index最小的根
                if 'index' not in dag_json[downstream_task_name]:
                    has_change = True
                    if 'index' in task:
                        dag_json[downstream_task_name]['index'] = task['index']
                else:
                    if task['index'] > dag_json[downstream_task_name]['index']:
                        has_change = True
                        dag_json[downstream_task_name]['index'] = task['index']

    # print(dag_json)
    # 根据上下行链路获取位置
    start_x = 50
    start_y = 50

    # 先把根的位置弄好，子节点多的排在左侧前方。

    # @pysnooper.snoop()
    def set_downstream_position(task_name):
        downstream_tasks = [x for x in dag_json[task_name]['downstream'] if dag_json[x]['index'] == dag_json[task_name]['index']]  # 获取相同树的下游节点
        downstream_tasks = sorted(downstream_tasks, key=lambda temp: dag_json[temp]['total_down_num'], reverse=True)  # 按子孙数目排序
        for i in range(len(downstream_tasks)):
            downstream_task = downstream_tasks[i]
            y = dag_json[downstream_task]['deep'] * 100 - 50
            # 获取前面的树有多少同一层叶子
            front_task_num = 0
            for temp in dag_json:
                # print(dag_json[temp]['index'],dag_json[task_name]['index'], dag_json[temp]['deep'],dag_json[task_name]['deep'])
                if dag_json[temp]['index'] < dag_json[downstream_task]['index'] and dag_json[temp]['deep'] == dag_json[downstream_task]['deep']:
                    front_task_num += 1
            front_task_num += i
            # y至少要操作他的上游节点的最小值。下游节点有多上上游节点时，靠左排布
            up = min([read_position(tasks[task_name]['id'])[0] for task_name in dag_json[downstream_task]['upstream']])  # 获取这个下游节点的全部上游节点的x值
            x = max(up,400*front_task_num+50)
            # x = 400*front_task_num+50
            set_position(str(tasks[downstream_task]['id']), x, y)

        # 布局下一层
        for temp in downstream_tasks:
            set_downstream_position(temp)

    # print(dag_json)
    # 一棵树一棵树的构建。优先布局下游叶子节点数量大的
    for task_name in root_nodes:
        task_id = str(tasks[task_name]['id'])
        set_position(task_id, start_x, start_y)
        start_x += 400
        set_downstream_position(task_name)

    return expand_tasks



def hive_create_sql_demo():
    sql = '''create table if not exists test_table(
    ftime int comment '分区时间',
    event_time string comment '事件时间戳'
    ) comment 'test'
    PARTITION BY LIST( ftime )
            (
                PARTITION p_20210925 VALUES IN ( 20210925 ),
                PARTITION default
            );
'''
    return sql


import subprocess


def run_shell(shell):
    cmd = subprocess.Popen(shell, stdin=subprocess.PIPE, stderr=sys.stderr, close_fds=True,
                           stdout=sys.stdout, universal_newlines=True, shell=True, bufsize=1)

    cmd.communicate()
    return cmd.returncode

# 将json的文本中的全部unicode_escape编码的文本转换回中文
def decode_unicode_escape(data):
    if isinstance(data, str):
        return data.encode('utf-8').decode('unicode_escape')
    elif isinstance(data, dict):
        return {decode_unicode_escape(key): decode_unicode_escape(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [decode_unicode_escape(item) for item in data]
    else:
        return data

# 获取一个host 配置中的url信息
def split_url(url):
    if not url:
        return '','',''
    if url[0]=='/':
        return '','',url
    if url[0]==':':
        url=url[1:]
        port = url.split("/")[0]
        return '',port,url[len(port):]
    url=url.replace('http://','').replace('https://','')
    if '/' in url:
        host_port = url[:url.index('/')]
        path = url.replace(host_port,'')
        if ':' in host_port:
            host,port = host_port.split(":")[0],host_port.split(":")[1]
        else:
            host, port = host_port,''
        return host,port,path
    else:
        if ':' in url:
            host,port = url.split(":")[0],url.split(":")[1]
        else:
            host, port = url,''
        return host,port,''


# @pysnooper.snoop()
def get_all_resource(cluster='all',namespace='all',exclude_pod=[]):
    import pandas
    from myapp.utils.py.py_k8s import K8s
    from myapp import conf
    if cluster=='all':
        clusters=conf.get('CLUSTERS')
    else:
        clusters={
            cluster:conf.get('CLUSTERS').get(cluster,{})
        }
    if namespace=='all':
        namespaces=conf.get('HUBSECRET_NAMESPACE',[])
    else:
        namespaces=[namespace]

    all_resource = []
    for cluser_name in clusters:
        cluster = clusters[cluser_name]
        k8s_client = K8s(cluster.get('KUBECONFIG', ''))
        for namespace in namespaces:
            pods = k8s_client.get_pods(namespace=namespace)
            for pod in pods:
                # 集群，资源组，空间，项目组，用户，resource，值
                user = pod['labels'].get('user', pod['labels'].get('username', pod['labels'].get('run-rtx',pod['labels'].get('rtx-user','admin'))))
                project = pod['annotations'].get('project', 'public')
                all_resource.append([cluser_name,pod['node_selector'].get('org','public'),namespace,project,user,pod['name'],pod['labels'],'cpu',float(pod['cpu'])])
                all_resource.append([cluser_name, pod['node_selector'].get('org', 'public'), namespace, project, user,pod['name'],pod['labels'], 'memory',float(pod['memory'])])
                gpu_resource = conf.get('GPU_RESOURCE', {})
                for ai_device in gpu_resource:
                    all_resource.append([cluser_name, pod['node_selector'].get('org', 'public'), namespace, project, user,pod['name'],pod['labels'], ai_device,float(pod.get(ai_device,''))])
    columns = ['cluster', 'org', 'namespace', 'project', 'user', 'name','labels','resource', 'value']
    all_resource =[dict(zip(columns,resource)) for resource in all_resource]
    # print(all_resource)
    if type(exclude_pod) == str:
        exclude_pod = [exclude_pod]
    # 直接指定排除的pod名称
    if type(exclude_pod)==list:
        all_resource = [pod for pod in all_resource if pod['name'] not in exclude_pod]
    # 通过字典过滤排除pod
    elif type(exclude_pod)==dict:
        all_resource = [pod for pod in all_resource if not all(item in pod['labels'].items() for item in exclude_pod.items())]
        # all_resource_temp = []
        # for pod in all_resource:
        #     can_add=False
        #     for key in exclude_pod:
        #         # 写的简洁些
        #         if key not in pod['labels'] or pod['labels'][key]!=exclude_pod[key]:
        #             can_add = True
        #             break
        #     if can_add:
        #         all_resource_temp.append(pod)
        # all_resource = all_resource_temp
    return all_resource

# 获取指定端口以后的5个非黑名单端口
def get_not_black_port(port):
    from myapp import conf
    black_port = conf.get('BLACK_PORT',[10250])
    meet_port = []
    while len(meet_port)<5:
        if port not in black_port:
            meet_port.append(port)
        port+=1
    return meet_port

# 验证用户资源额度限制
@pysnooper.snoop()
def meet_quota(req_user,req_project,req_cluster_name,req_org,req_namespace,exclude_pod=[],req_resource={},replicas=1):
    # 管理员不受限制
    if req_user.is_admin():
        return True,''
    # resource为{"cpu":1,"memory":1,"gpu":1}格式
    # quota 书写格式，cluster_name，org,namespace，resource，single_total,value
    # exclude_pod数组格式表示忽略的名称数组，字典格式表述忽略的pod标签，字符串表示原始的pod名

    # 添加对gpu型号的处理
    req_resource={
        "cpu": str(req_resource.get('cpu','0')),
        "memory": str(req_resource.get('memory','0')),
        "gpu": str(req_resource.get('gpu','0')),
    }
    if '(' in req_resource['gpu']:
        req_resource['gpu'] = req_resource['gpu'][:req_resource['gpu'].index('(')]

    all_resources = None
    # req_total_resource={key:float(str(req_resource[key]).replace('G',''))*replicas for key in req_resource}
    # 验证用户username在集群cluster_name的namespace空闲下运行value大的resource(cpu,memory,gpu)资源是否允许
    # 先来验证是否有个人用户额度限制，单集群限制，单空闲限制
    if req_user.quota:
        if not all_resources:
            all_resources = get_all_resource(exclude_pod=exclude_pod)

        quota_confg = req_user.quota
        quotas_array = re.split(';|\n',quota_confg.strip())
        quotas=[]
        for quota in quotas_array:
            quota = quota.replace(' ','').strip()
            if len(quota.split(','))==6:
                quotas.append(dict(zip(['cluser','org','namespace','resource','type','value'],quota.split(','))))

        for quota in quotas:
            if quota['namespace']=='notebook':
                quota['namespace']='jupyter'

            # 查看单个任务是否满足资源限制
            if quota['type']=='single':
                if req_cluster_name == quota['cluser'] or quota['cluser'] == 'all':
                    if req_org == quota['org'] or quota['org'] == 'all':
                        if req_namespace==quota['namespace'] or quota['namespace']=='all':
                            limit_resource = float(quota['value'])
                            request_resource = float(str(req_resource.get(quota['resource'],'0')).replace('G',''))
                            message = f'user {quota["type"]} quota: \nrequest {quota["resource"]} {request_resource}, user limit {limit_resource}'
                            print(message)
                            if request_resource>limit_resource:
                                return False,Markup("<br>"+message.replace('\n','<br>'))

            # 查看正在运行的整体资源是否满足资源限制
            if quota['type']=='concurrent':
                if req_cluster_name == quota['cluser'] or quota['cluser'] == 'all':
                    if req_org == quota['org'] or quota['org'] == 'all':
                        if req_namespace==quota['namespace'] or quota['namespace']=='all':
                            exist_pod = all_resources
                            # 过滤个人名下的pod
                            exist_pod = [pod for pod in exist_pod if pod['user'] == req_user.username]

                            if quota['cluser']!='all':
                                exist_pod = [pod for pod in exist_pod if pod['cluster']==quota['cluser']]
                            if quota['namespace']!='all':
                                exist_pod = [pod for pod in exist_pod if pod['namespace'] == quota['namespace']]
                            if quota['org']!='all':
                                exist_pod = [pod for pod in exist_pod if pod['org'] == quota['org']]

                            exist_resource = sum([float(str(pod.get('value','0')).replace('G','')) for pod in exist_pod])
                            limit_resource = float(quota['value'])
                            request_resource = float(str(req_resource.get(quota['resource'], '0')).replace('G', ''))
                            message = f'user {quota["type"]} quota: \nrequest {quota["resource"]} {request_resource} * {replicas}, user limit {limit_resource}, exist {exist_resource}'
                            message += "\nexist pod:\n" + '\n'.join([pod['labels'].get('pod-type', 'task') + ":" + pod['name'] for pod in exist_pod])

                            print(message)
                            if request_resource*replicas>(limit_resource-exist_resource):
                                return False,Markup("<br>"+message.replace('\n','<br>'))

            if quota['type']=='total':
                pass

    # 或者这个项目下的限制，比如对每个人的限制和对项目组的总和设置，已经确保了申请项目组，与额度配置项目组相同
    if req_project.quota():
        if not all_resources:
            all_resources = get_all_resource(exclude_pod=exclude_pod)

        quota_confg = req_project.quota()
        quotas_array = re.split(';|\n', quota_confg.strip())
        quotas = []
        for quota in quotas_array:
            if len(quota.split(',')) == 4:
                quota = quota.replace(' ', '').strip()
                quotas.append(dict(zip(['namespace', 'resource', 'type', 'value'], quota.split(','))))

        for quota in quotas:
            if quota['namespace']=='notebook':
                quota['namespace']='jupyter'

            # 查看单个任务是否满足资源限制
            if quota['type'] == 'single':
                if req_namespace==quota['namespace'] or quota['namespace']=='all':
                    limit_resource = float(quota['value'])
                    request_resource = float(str(req_resource.get(quota['resource'], '0')).replace('G', ''))
                    message = f'project {quota["type"]} quota: \nrequest {quota["resource"]} {request_resource}, project limit {limit_resource}'
                    print(message)
                    if request_resource > limit_resource:
                        return False,Markup("<br>"+message.replace('\n','<br>'))

            # 查看正在运行的整体资源是否满足资源限制
            if quota['type'] == 'concurrent':
                if req_namespace == quota['namespace'] or quota['namespace'] == 'all':
                    exist_pod = all_resources
                    # 过滤该项目组下的pod
                    exist_pod = [pod for pod in exist_pod if pod['project'] == req_project.name]

                    # print(exist_pod)
                    if quota['namespace'] != 'all':
                        exist_pod = [pod for pod in exist_pod if pod['namespace'] == quota['namespace']]

                    exist_pod = [pod for pod in exist_pod if pod['resource'] == quota['resource']]
                    # print(exist_pod)
                    # print(quota['resource'])
                    exist_resource = sum([float(str(pod.get('value', '0')).replace('G', '')) for pod in exist_pod])
                    limit_resource = float(quota['value'])
                    request_resource = float(str(req_resource.get(quota['resource'], '0')).replace('G', ''))
                    message = f'project {quota["type"]} quota: \nrequest {quota["resource"]} {request_resource} * {replicas}, project limit {limit_resource}, exist {exist_resource}'
                    message +="\nexist pod:\n"+'\n'.join([pod['labels'].get('pod-type','task')+":"+pod['name'] for pod in exist_pod])
                    print(message)
                    # message += '\n<a target="_blank" href="https://www.w3schools.com">申请资源</a>'
                    if request_resource*replicas > (limit_resource - exist_resource):
                        return False,Markup("<br>"+message.replace('\n','<br>'))

            if quota['type'] == 'total':
                pass

    # 获取项目组下对个人的额度限制，已经确保了申请项目组，与额度配置项目组相同
    if req_project.quota(userid=req_user.id):

        if not all_resources:
            all_resources = get_all_resource(exclude_pod=exclude_pod)

        quota_confg = req_project.quota(userid=req_user.id)
        quotas_array = re.split(';|\n', quota_confg.strip())
        quotas = []
        for quota in quotas_array:
            quota = quota.replace(' ', '').strip()
            if len(quota.split(',')) == 4:
                quotas.append(dict(zip(['namespace', 'resource', 'type', 'value'], quota.split(','))))

        for quota in quotas:
            if quota['namespace']=='notebook':
                quota['namespace']='jupyter'

            # 查看单个任务是否满足资源限制
            if quota['type'] == 'single':
                if req_namespace == quota['namespace'] or quota['namespace'] == 'all':
                    limit_resource = float(quota['value'])
                    request_resource = float(str(req_resource.get(quota['resource'], '0')).replace('G', ''))
                    message = f'project user {quota["type"]} quota: \nrequest {quota["resource"]} {request_resource}, project limit {limit_resource}'
                    print(message)
                    if request_resource > limit_resource:
                        return False,Markup("<br>"+message.replace('\n','<br>'))

            # 查看正在运行的整体资源是否满足资源限制，
            if quota['type'] == 'concurrent':
                if req_namespace == quota['namespace'] or quota['namespace'] == 'all':
                    exist_pod = all_resources
                    # 过滤该项目组下的pod
                    exist_pod = [pod for pod in exist_pod if pod['project'] == req_project.name and pod['user']==req_user.username]
                    if quota['namespace'] != 'all':
                        exist_pod = [pod for pod in exist_pod if pod['namespace'] == quota['namespace']]

                    exist_pod = [pod for pod in exist_pod if pod['resource'] == quota['resource']]

                    # print(exist_pod)
                    # print(quota['resource'])
                    exist_resource = sum([float(str(pod.get('value', '0')).replace('G', '')) for pod in exist_pod])
                    limit_resource = float(quota['value'])
                    request_resource = float(str(req_resource.get(quota['resource'], '0')).replace('G', ''))
                    message = f'project user {quota["type"]} quota: \nrequest {quota["resource"]} {request_resource} * {replicas}, project limit {limit_resource}, exist {exist_resource}'
                    message +="\nexist pod:\n"+'\n'.join([pod['labels'].get('pod-type','task')+":"+pod['name'] for pod in exist_pod])

                    print(message)
                    if request_resource*replicas > (limit_resource - exist_resource):
                        return False,Markup("<br>"+message.replace('\n','<br>'))

            if quota['type'] == 'total':
                pass

    return True,''


def test_database_connection(url):
    from sqlalchemy import create_engine
    from sqlalchemy.exc import OperationalError
    try:
        engine = create_engine(url)
        conn = engine.connect()
        conn.close()
        return True
    except OperationalError:
        return False