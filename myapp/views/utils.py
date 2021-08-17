
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from urllib import parse

from flask import request
import simplejson as json

from myapp import app, db
from myapp.exceptions import MyappException

import time


FORM_DATA_KEY_BLACKLIST: List[str] = []
if not app.config.get("ENABLE_JAVASCRIPT_CONTROLS"):
    FORM_DATA_KEY_BLACKLIST = ["js_tooltip", "js_onclick_href", "js_data_mutator"]


# 获取用户信息
def bootstrap_user_data(user, include_perms=False):

    payload = {
        "username": user.username,
        "firstName": user.first_name,
        "lastName": user.last_name,
        "userId": user.id,
        "isActive": user.is_active,
        "createdOn": user.created_on.isoformat(),
        "email": user.email,
    }

    if include_perms:
        roles, permissions = get_permissions(user)
        payload["roles"] = roles
        payload["permissions"] = permissions

    return payload

# 获取用户权限
def get_permissions(user):
    if not user.roles:
        raise AttributeError("User object does not have roles")

    roles = {}
    permissions = defaultdict(set)
    for role in user.roles:
        perms = set()
        for perm in role.permissions:
            if perm.permission and perm.view_menu:
                perms.add((perm.permission.name, perm.view_menu.name))
                if perm.permission.name in ("datasource_access", "database_access"):
                    permissions[perm.permission.name].add(perm.view_menu.name)
        roles[role.name] = [
            [perm.permission.name, perm.view_menu.name]
            for perm in role.permissions
            if perm.permission and perm.view_menu
        ]

    return roles, permissions


