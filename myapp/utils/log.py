from abc import ABC, abstractmethod
import datetime
import functools
import json

from flask import current_app, g, request


# @pysnooper.snoop(depth=2)
class AbstractEventLogger(ABC):
    @abstractmethod
    def log(self, user_id, action, duration_ms, *args, **kwargs):
        pass

    def log_this(self, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            user_id = None
            if g.user:
                user_id = g.user.get_id()
            d = request.form.to_dict() or {}

            # request parameters can overwrite post body
            request_params = request.args.to_dict()
            d.update(request_params)
            d.update(kwargs)

            d.update(request.json or {})

            self.stats_logger.incr(f.__name__)
            start_dttm = datetime.datetime.now()
            value = f(*args, **kwargs)
            duration_ms = (datetime.datetime.now() - start_dttm).total_seconds() * 1000

            if user_id:
                self.log(
                    user_id,
                    f.__name__,
                    duration_ms=duration_ms,
                    **d
                )
            return value

        return wrapper

    @property
    def stats_logger(self):
        return current_app.config.get("STATS_LOGGER")


# @pysnooper.snoop(depth=2)
class DBEventLogger(AbstractEventLogger):

    def log(self, user_id, action, duration_ms, *args, **kwargs):
        from myapp.models.log import Log
        referrer = request.referrer[:1000] if request.referrer else None

        log = Log(
            action=action,
            json=json.dumps(kwargs, indent=4, ensure_ascii=False),
            duration_ms=duration_ms,
            referrer=referrer,
            user_id=user_id,
            method=request.method,
            path=request.path,
            dttm=datetime.datetime.now()
        )

        sesh = current_app.appbuilder.get_session
        sesh.add(log)
        sesh.commit()
