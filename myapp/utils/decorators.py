
from datetime import datetime, timedelta
from functools import wraps
import logging

from contextlib2 import contextmanager
from flask import request

from myapp import app
from myapp.utils.dates import now_as_float




@contextmanager
def stats_timing(stats_key, stats_logger):
    """Provide a transactional scope around a series of operations."""
    start_ts = now_as_float()
    try:
        yield start_ts
    except Exception as e:
        raise e
    finally:
        stats_logger.timing(stats_key, now_as_float() - start_ts)


def etag(check_perms=bool):

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # check if the user can access the resource
            check_perms(*args, **kwargs)

            if request.method == "POST":
                return f(*args, **kwargs)

            response = None

            if response is None:
                response = f(*args, **kwargs)

            return response.make_conditional(request)

        return wrapper

    return decorator
