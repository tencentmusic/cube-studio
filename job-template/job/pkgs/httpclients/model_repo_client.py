
import requests
import traceback
import json
from datetime import datetime
from ..constants import ModelStatus
from ..exceptions.model_repo_exception import *
from ..context import KFJobContext

API_BASE_URL = "http://kubeflow-dashboard.infra"
INNER_API_BASE_URL = "http://kubeflow-dashboard.infra"
MODEL_API_URI = "training_model_modelview/api"
DEPLOY_API_URI = "training_model_deploy_modelview/api"
EMBEDDING_API_URI = "embedding_modelview/api"


class ModelRepoClient(object):
    def __init__(self, auth_user):
        self.auth_user = auth_user
        self.api_base_url = KFJobContext.get_context().model_repo_api_url or INNER_API_BASE_URL

    def add_model(self, pipeline_id, run_id, framework, model_name, model_path, model_version,
                  model_desc, thr_exp=True):
        api_url = self._get_model_api_url() + "/"
        model_info = {
            "pipeline": pipeline_id,
            "run_id": run_id,
            "framework": framework,
            "name": model_name,
            "path": model_path,
            "describe": model_desc,
            "version": model_version,
            "status": ModelStatus.OFFLINE,
            # "return_type": "json",
            'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        try:
            resp = self._request('post', api_url, data=model_info)
            if resp.status_code // 100 > 2:
                print("add model info to repository http failed, api_url='{}', model_info={}, resp=[{}, {}, {}]"
                      .format(api_url, model_info, resp.status_code, resp.reason, resp.text))
                if thr_exp:
                    raise AddModelException("add model http error: [{}, {}, {}]"
                                            .format(resp.status_code, resp.reason, resp.text))
                return None
            try:
                resp_body = json.loads(resp.text)
            except Exception as e1:
                print("load response as json failed, api_url='{}', model_info={}, resp=[{}, {}, {}]"
                      .format(api_url, model_info, resp.status_code, resp.reason, resp.text))
                if thr_exp:
                    raise e1
                return None
            if resp_body.get('status') != 0:
                print("add model info to repository server failed, api_url='{}', model_info={}, resp_body={}"
                      .format(api_url, model_info, resp_body))
                if thr_exp:
                    raise AddModelException("add model server error: {}".format(resp_body))
                return None
            print("added model to repository: {}".format(resp_body['result']))
            return resp_body['result']
        except Exception as e:
            print("add model info to repository error, api_url='{}': {}\nmodel_info={}\n{}"
                  .format(api_url, e, model_info, traceback.format_exc()))
            if thr_exp:
                raise e
            return None

    def update_model(self, pipeline_id, run_id, updates, thr_exp=True):
        if not updates:
            print("no field to update to model of pipeline_id '{}' of run_id '{} in repository"
                  .format(pipeline_id, run_id))
            return None
        model_info = self.query_model_by_id(pipeline_id, run_id, thr_exp)
        if not model_info:
            print("found no model of pipeline_id '{}' of run_id '{} from repository', can not update"
                  .format(pipeline_id, run_id))
            return None
        print("found model of pipeline_id '{}' of run_id '{} in repository: {}".format(pipeline_id, run_id, model_info))
        model_id = model_info.get('id')
        api_url = self._get_model_api_url(model_id)

        model_info.update(updates)
        [model_info.pop(key) for key in ['id', 'pipeline_id', 'changed_by_fk', 'changed_on',
                                         'created_by_fk', 'created_on'] if key in model_info]
        model_info['pipeline'] = pipeline_id
        # model_info['return_type'] = 'json'
        try:
            resp = self._request('put', api_url, data=model_info)
            if resp.status_code // 100 > 2:
                print("update model to repository http failed, api_url='{}', updates={}, model_info={},"
                      " resp=[{}, {}, {}]".format(api_url, updates, model_info, resp.status_code,
                                                  resp.reason, resp.text))
                if thr_exp:
                    raise UpdateModelException("update model http error: [{}, {}, {}]"
                                               .format(resp.status_code, resp.reason, resp.text))
                return None
            try:
                resp_body = json.loads(resp.text)
            except Exception as e1:
                print("load response as json failed, api_url='{}', model_info={}, resp=[{}, {}, {}]"
                      .format(api_url, model_info, resp.status_code, resp.reason, resp.text))
                if thr_exp:
                    raise e1
                return None
            if resp_body.get('status') != 0:
                print("update model to repository server failed, api_url='{}', updates={}, model_info={}, resp_body={}"
                      .format(api_url, updates, model_info, resp_body))
                if thr_exp:
                    raise UpdateModelException("update model server error: {}".format(resp_body))
                return None
            return resp_body['result']
        except Exception as e:
            print("update model to repository error, api_url='{}': {}\nupdates={}, model_info={}\n{}"
                  .format(api_url, e, updates, model_info, traceback.format_exc()))
            if thr_exp:
                raise e
            return None

    def query_model_by_id(self, pipeline_id, run_id, thr_exp=True):
        api_url = self._get_model_api_url() + "/"
        # params = {
        #     "_flt_0_pipeline_id": pipeline_id,
        #     "_flt_3_run_id": run_id
        # }
        # data = {
        #     "return_type": "json"
        # }
        payload = {
            "page": 0,
            "page_size": 10,
            "filters": [
                {"col": "pipeline", "opr": "rel_o_m", "value": pipeline_id},
                {"col": "run_id", "opr": "eq", "value": run_id}
            ]
        }
        try:
            resp = self._request('get', api_url, data=payload)
            if resp.status_code // 100 > 2:
                print("query model from repository http failed, api_url='{}', payload={}, resp=[{}, {}, {}]"
                      .format(api_url, payload, resp.status_code, resp.reason, resp.text))
                if thr_exp:
                    raise QueryModelException("query model from repository http error: [{}, {}, {}]"
                                              .format(resp.status_code, resp.reason, resp.text))
                return None
            try:
                resp_body = json.loads(resp.text)
            except Exception as e1:
                print("load response as json failed, api_url='{}', resp=[{}, {}, {}]"
                      .format(api_url, resp.status_code, resp.reason, resp.text))
                if thr_exp:
                    raise e1
                return None
            if resp_body.get('status') != 0:
                print("query model from repository server failed, api_url='{}', payload={}, resp_body={}"
                      .format(api_url, payload, resp_body))
                if thr_exp:
                    raise QueryModelException("query model from repository server error: {}".format(resp_body))
                return None
            model_infos = resp_body.get('result')
            if not model_infos:
                print("found no model from repository, api_url='{}', payload={}, resp_body={}"
                      .format(api_url, payload, resp_body))
                return None
            if not isinstance(model_infos, list):
                print("invalid model list from repository, api_url='{}', payload={}, resp_body={}"
                      .format(api_url, payload, resp_body))
                if thr_exp:
                    raise QueryModelException("invalid model list from repository: {}".format(resp_body))
                return None
            return model_infos[0]
        except Exception as e:
            print("query model error, api_url='{}': {}\nparams={}\n{}"
                  .format(api_url, e, payload, traceback.format_exc()))
            if thr_exp:
                raise e
            return None

    def query_models_by_status(self, pipeline_id, status, thr_exp=True):
        api_url = self._get_model_api_url() + "/"
        # params = {
        #     "_flt_0_pipeline_id": pipeline_id,
        #     "_flt_0_status": status
        # }
        # data = {
        #     "return_type": "json"
        # }
        payload = {
            "page": 0,
            "page_size": 10,
            "filters": [
                {"col": "status", "opr": "eq", "value": status}
            ]
        }
        try:
            resp = self._request('get', api_url, data=payload)
            if resp.status_code // 100 > 2:
                print("query model from repository http failed, api_url='{}', payload={}, resp=[{}, {}, {}]"
                      .format(api_url, payload, resp.status_code, resp.reason, resp.text))
                if thr_exp:
                    raise QueryModelException("query model from repository http error: [{}, {}, {}]"
                                              .format(resp.status_code, resp.reason, resp.text))
                return None
            try:
                resp_body = json.loads(resp.text)
            except Exception as e1:
                print("load response as json failed, api_url='{}', resp=[{}, {}, {}]"
                      .format(api_url, resp.status_code, resp.reason, resp.text))
                if thr_exp:
                    raise e1
                return None
            if resp_body.get('status') != 0:
                print("query model from repository server failed, api_url='{}', payload={}, resp_body={}"
                      .format(api_url, payload, resp_body))
                if thr_exp:
                    raise QueryModelException("query model from repository server error: {}".format(resp_body))
                return None
            model_infos = resp_body.get('result')
            if not model_infos:
                print("found no model from repository, api_url='{}', payload={}, resp_body={}"
                      .format(api_url, payload, resp_body))
                return []
            if not isinstance(model_infos, list):
                print("invalid model list from repository, api_url='{}', payload={}, resp_body={}"
                      .format(api_url, payload, resp_body))
                if thr_exp:
                    raise QueryModelException("invalid model list from repository: {}".format(resp_body))
                return None
            return model_infos
        except Exception as e:
            print("query model error, api_url='{}': {}\npayload={}\n{}"
                  .format(api_url, e, payload, traceback.format_exc()))
            if thr_exp:
                raise e
            return None

    def get_online_model_info(self, pipeline_id, thr_exp=True):
        model_infos = self.query_models_by_status(pipeline_id, ModelStatus.ONLINE, thr_exp)
        if not model_infos:
            print("found no online model of pipeline_id '{}'".format(pipeline_id))
            return None
        if len(model_infos) > 1:
            model_infos = sorted(model_infos, key=lambda x: x['changed_on'], reverse=True)
            print("WARNING: found {} online models of pipeline_id '{}', will choose the last changed model: {}"
                  .format(len(model_infos), pipeline_id, model_infos))
        return model_infos[0]

    def deploy_model(self, env, model_project_name, model_name, model_path, thr_exp=True):
        env = 'prod' if env in ['prod', 'product'] else 'test'
        api_url = self._get_model_api_url('deploy', env)
        deploy_info = {
            "model_project_name": model_project_name,
            "model_name": model_name,
            "model_path": model_path
        }
        try:
            resp = self._request('post', api_url, data=deploy_info)
            if resp.status_code // 100 > 2:
                print("deploy model http failed, api_url='{}', deploy_info={}, resp=[{}, {}, {}]"
                      .format(api_url, deploy_info, resp.status_code, resp.reason, resp.text))
                raise DeployModelException("deploy model http error: [{}, {}, {}]"
                                           .format(resp.status_code, resp.reason, resp.text))
            try:
                resp_body = json.loads(resp.text)
            except Exception as e1:
                print("load response as json failed, api_url='{}', deploy_info={}, resp=[{}, {}, {}]"
                      .format(api_url, deploy_info, resp.status_code, resp.reason, resp.text))
                raise e1
            if resp_body.get('status') != 0:
                print("deploy model server failed, api_url='{}', deploy_info={}, resp_body={}"
                      .format(api_url, deploy_info, resp_body))
                raise DeployModelException("deploy model server error: {}".format(resp_body))
            print("deployed model under env '{}': {}".format(env, resp_body['result']))
            return resp_body['result']
        except Exception as e:
            print("deploy model error, api_url='{}': {}\ndeploy_info={}\n{}"
                  .format(api_url, e, deploy_info, traceback.format_exc()))
            if thr_exp:
                raise e
            return None

    def query_embedding_versions_by_model(self, project_name, model_name, thr_exp=True):
        api_url = self._get_embedding_api_url() + "/"
        payload = {
            "page": 0,
            "page_size": 1000000000,
            "filters": [
                {"col": "project", "opr": "rel_o_m", "value": project_name},
                {"col": "model_name", "opr": "eq", "value": model_name}
            ]
        }
        try:
            resp = self._request('get', api_url, data=payload)
            if resp.status_code // 100 > 2:
                print("query embedding versions http failed, api_url='{}', payload={}, resp=[{}, {}, {}]"
                      .format(api_url, payload, resp.status_code, resp.reason, resp.text))
                raise QueryEmbeddingException("query embedding versions http error: [{}, {}, {}]"
                                              .format(resp.status_code, resp.reason, resp.text))
            try:
                resp_body = json.loads(resp.text)
            except Exception as e1:
                print("load response as json failed, api_url='{}', resp=[{}, {}, {}]"
                      .format(api_url, resp.status_code, resp.reason, resp.text))
                raise e1
            if resp_body.get('status') != 0:
                print("query embedding versions server failed, api_url='{}', payload={}, resp_body={}"
                      .format(api_url, payload, resp_body))
                raise QueryEmbeddingException("query embedding versions server error: {}".format(resp_body))
            emb_vers = resp_body.get('result')
            if not isinstance(emb_vers, list) or not emb_vers:
                print("found no embedding versions from repository, api_url='{}', payload={}, resp_body={}"
                      .format(api_url, payload, resp_body))
                return []
            return emb_vers
        except Exception as e:
            print("query embedding versions error, api_url='{}': {}\nparams={}\n{}"
                  .format(api_url, e, payload, traceback.format_exc()))
            if thr_exp:
                raise e
            return None

    def add_embedding_version(self, pipeline_id, run_id, project_name, model_name, version, is_fallback, metrics,
                              model_path, embedding_file_path, thr_exp=True):
        api_url = self._get_embedding_api_url() + "/"
        embedding_info = {
            "project": project_name,
            "model_name": model_name,
            "version": version,
            "is_fallback": is_fallback,
            "metrics": json.dumps(metrics),
            "pipeline": pipeline_id,
            "run_id": run_id,
            "model_path": model_path,
            "embedding_file_path": embedding_file_path
        }
        try:
            resp = self._request('post', api_url, data=embedding_info)
            if resp.status_code // 100 > 2:
                print("add embedding version to repository http failed, api_url='{}', embedding_info={},"
                      " resp=[{}, {}, {}]".format(api_url, embedding_info, resp.status_code, resp.reason, resp.text))
                raise AddEmbeddingException("add embedding version http error: [{}, {}, {}]"
                                            .format(resp.status_code, resp.reason, resp.text))
            try:
                resp_body = json.loads(resp.text)
            except Exception as e1:
                print("load response as json failed, api_url='{}', embedding_info={}, resp=[{}, {}, {}]"
                      .format(api_url, embedding_info, resp.status_code, resp.reason, resp.text))
                raise e1
            if resp_body.get('status') != 0:
                print("add embedding version to repository server failed, api_url='{}', embedding_info={}, resp_body={}"
                      .format(api_url, embedding_info, resp_body))
                raise AddEmbeddingException("add embedding version server error: {}".format(resp_body))
            print("added embedding version to repository: {}".format(embedding_info))
            return resp_body['result']
        except Exception as e:
            print("add embedding version to repository error, api_url='{}': {}\nembedding_info={}\n{}"
                  .format(api_url, e, embedding_info, traceback.format_exc()))
            if thr_exp:
                raise e
            return None

    def delete_embedding_version(self, vid, thr_exp=True):
        api_url = self._get_embedding_api_url(vid)
        try:
            resp = self._request('delete', api_url)
            if resp.status_code // 100 > 2:
                print("delete embedding version http failed, api_url='{}' resp=[{}, {}, {}]"
                      .format(api_url, resp.status_code, resp.reason, resp.text))
                raise DeleteEmbeddingException("delete embedding version http error: [{}, {}, {}]"
                                               .format(resp.status_code, resp.reason, resp.text))
            try:
                resp_body = json.loads(resp.text)
            except Exception as e1:
                print("load response as json failed, api_url='{}', resp=[{}, {}, {}]"
                      .format(api_url, resp.status_code, resp.reason, resp.text))
                raise e1
            if resp_body.get('status') != 0:
                print("delete embedding version server failed, api_url='{}', resp_body={}".format(api_url, resp_body))
                raise DeleteEmbeddingException("delete embedding version server error: {}".format(resp_body))
            print("deleted embedding version '{}': {}".format(vid, resp_body))
            return 0
        except Exception as e:
            print("delete embedding version error, api_url='{}': {}\n{}".format(api_url, e, traceback.format_exc()))
            if thr_exp:
                raise e
            return -1

    def _get_model_api_url(self, *postfixs):
        return '/'.join([self.api_base_url, MODEL_API_URI] + list(map(lambda x: str(x), postfixs)))

    def _get_deploy_api_url(self, *postfixs):
        return '/'.join([self.api_base_url, DEPLOY_API_URI] + list(map(lambda x: str(x), postfixs)))

    def _get_embedding_api_url(self, *postfixs):
        return '/'.join([self.api_base_url, EMBEDDING_API_URI] + list(map(lambda x: str(x), postfixs)))

    def _make_headers(self):
        return {"Content-Type": "application/json", "Authorization": self.auth_user}

    def _request(self, method, url, params=None, data=None):
        return requests.request(method, url, params=params, json=data, headers=self._make_headers())


