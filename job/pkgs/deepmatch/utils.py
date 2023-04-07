# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

import json
import logging
from threading import Thread

import requests

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse


import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda


def recall_N(y_true, y_pred, N=50):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)


def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)

def get_item_embedding(item_embedding, item_input_layer):
    return Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)



def check_version(version):
    """Return version of package on pypi.python.org using json."""

    def check(version):
        try:
            url_pattern = 'https://pypi.python.org/pypi/deepmatch/json'
            req = requests.get(url_pattern)
            latest_version = parse('0')
            version = parse(version)
            if req.status_code == requests.codes.ok:
                j = json.loads(req.text.encode('utf-8'))
                releases = j.get('releases', [])
                for release in releases:
                    ver = parse(release)
                    if ver.is_prerelease or ver.is_postrelease:
                        continue
                    latest_version = max(latest_version, ver)
                if latest_version > version:
                    logging.warning(
                        '\nDeepMatch version {0} detected. Your version is {1}.\nUse `pip install -U deepmatch` to upgrade.Changelog: https://github.com/shenweichen/DeepMatch/releases/tag/v{0}'.format(
                            latest_version, version))
        except:
            print("Please check the latest version manually on https://pypi.org/project/deepmatch/#history")
            return

    Thread(target=check, args=(version,)).start()
