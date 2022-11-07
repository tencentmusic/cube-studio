from abc import ABC, abstractmethod
import datetime
import functools
import json
import uuid
from flask import current_app, g, request,jsonify,session

all_history={

}

# @pysnooper.snoop(depth=2)
class AbstractEventLogger(ABC):

    def log_this(f):
        def wraps(*args, **kwargs):
            username = "anonymous-" + uuid.uuid4().hex[:16]
            if session.get('username', ''):
                username = session.get('username', '')
            req_url = request.path
            print(req_url)
            num = all_history.get(username, {}).get(req_url, 0)
            if num > 1 and '/api/model/' in req_url and 'anonymous-' in username:
                return jsonify({
                    "status": 1,
                    "result": {},
                    "message": "匿名用户尽可访问一次，获得更多访问次数，需登录并激活用户"
                })
            res = f(*args, **kwargs)
            # # 登记推理日志
            # if '/info' in req_url:
            #     res.set_cookie('username', username)

            all_history[username] = {
                req_url: all_history.get(username, {}).get(req_url, 0) + 1
            }
            print(all_history)
            return res

        return wraps

