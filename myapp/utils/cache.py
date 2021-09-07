from flask import request

def view_cache_key(*unused_args, **unused_kwargs) -> str:
    args_hash = hash(frozenset(request.args.items()))
    return "view/{}/{}".format(request.path, args_hash)


def memoized_func(key=view_cache_key, attribute_in_key=None):
    """Use this decorator to cache functions that have predefined first arg.

    enable_cache is treated as True by default,
    except enable_cache = False is passed to the decorated function.

    force means whether to force refresh the cache and is treated as False by default,
    except force = True is passed to the decorated function.

    timeout of cache is set to 600 seconds by default,
    except cache_timeout = {timeout in seconds} is passed to the decorated function.

    memoized_func uses simple_cache and stored the data in memory.
    Key is a callable function that takes function arguments and
    returns the caching key.
    """
    def wrap(f):
        # noop
        def wrapped_f(self, *args, **kwargs):
            return f(self, *args, **kwargs)

        return wrapped_f

    return wrap
