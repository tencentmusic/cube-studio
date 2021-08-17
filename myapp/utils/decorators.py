
from datetime import datetime, timedelta
from functools import wraps
import logging

from contextlib2 import contextmanager
from flask import request

from myapp import app, cache
from myapp.utils.dates import now_as_float


# If a user sets `max_age` to 0, for long the browser should cache the
# resource? Flask-Caching will cache forever, but for the HTTP header we need
# to specify a "far future" date.
FAR_FUTURE = 365 * 24 * 60 * 60  # 1 year in seconds


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
    """
    A decorator for caching views and handling etag conditional requests.

    The decorator adds headers to GET requests that help with caching: Last-
    Modified, Expires and ETag. It also handles conditional requests, when the
    client send an If-Matches header.

    If a cache is set, the decorator will cache GET responses, bypassing the
    dataframe serialization. POST requests will still benefit from the
    dataframe cache for requests that produce the same SQL.

    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # check if the user can access the resource
            check_perms(*args, **kwargs)

            # for POST requests we can't set cache headers, use the response
            # cache nor use conditional requests; this will still use the
            # dataframe cache in `myapp/viz.py`, though.
            if request.method == "POST":
                return f(*args, **kwargs)

            response = None

            # if no response was cached, compute it using the wrapped function
            if response is None:
                response = f(*args, **kwargs)

            return response.make_conditional(request)

        return wrapper

    return decorator
