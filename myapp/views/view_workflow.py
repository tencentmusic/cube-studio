import copy
import os
import re
import time

import flask
from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from myapp.utils import core
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask import jsonify, make_response, send_from_directory, send_file
import pysnooper
from myapp.models.model_job import Pipeline, Workflow
from flask_appbuilder.actions import action
from myapp.project import push_message
from myapp import app, appbuilder, db, event_logger, cache
from flask import request
from sqlalchemy import or_
from flask import Markup
from myapp.utils.py import py_k8s
import logging
from .baseApi import (
    MyappModelRestApi
)
from flask import (
    abort,
    flash,
    g,
    redirect
)

from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)
from flask_appbuilder import expose
import datetime, json

conf = app.config


class CRD_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query.order_by(self.model.create_time.desc())
        return query.filter(
            or_(
                self.model.labels.contains('"%s"' % g.user.username),
            )
        ).order_by(self.model.create_time.desc())


class Crd_ModelView_Base():
    list_columns = ['name', 'namespace_url', 'create_time', 'status', 'username', 'stop']
    show_columns = ['name', 'namespace', 'create_time', 'status', 'annotations', 'labels', 'spec', 'status_more', 'info_json_html']
    order_columns = ['id']
    base_permissions = ['can_show', 'can_list', 'can_delete']
    # base_permissions = ['list','delete','show']
    crd_name = ''
    base_order = ('create_time', 'desc')
    base_filters = [["id", CRD_Filter, lambda: []]]

    # list
    def base_list(self):
        k8s_client = py_k8s.K8s()
        crd_info = conf.get("CRD_INFO", {}).get(self.crd_name, {})
        if crd_info:
            crds = k8s_client.get_crd_all_namespaces(group=crd_info['group'],
                                                     version=crd_info['version'],
                                                     plural=crd_info['plural'])

            # 删除所有，注意最好id从0开始
            db.session.query(self.datamodel.obj).delete()
            # db.engine.execute("alter table %s auto_increment =0"%self.datamodel.pbj.__tablename__)
            # 添加记录
            for crd in crds:
                try:
                    labels = json.loads(crd['labels'])
                    if 'run-rtx' in labels:
                        crd['username'] = labels['run-rtx']
                    elif 'pipeline-rtx' in labels:
                        crd['username'] = labels['pipeline-rtx']
                except Exception as e:
                    logging.error(e)
                crd_model = self.datamodel.obj(**crd)
                db.session.add(crd_model)

            db.session.commit()

    # 个性化删除操作
    def delete_more(self, item):
        pass

    # 基础批量删除
    # @pysnooper.snoop()
    def base_muldelete(self, items):
        if not items:
            abort(404)
        for item in items:
            self.delete_more(item)
            if item:
                try:
                    labels = json.loads(item.labels) if item.labels else {}
                    kubeconfig = None
                    if 'pipeline-id' in labels:
                        pipeline = db.session.query(Pipeline).filter_by(id=int(labels['pipeline-id'])).first()
                        if pipeline:
                            kubeconfig = pipeline.project.cluster.get('KUBECONFIG', '')

                    k8s_client = py_k8s.K8s(kubeconfig)
                    crd_info = conf.get("CRD_INFO", {}).get(self.crd_name, {})
                    if crd_info:
                        k8s_client.delete_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=item.namespace,name=item.name)
                        # db_crds = db.session.query(self.datamodel.obj).filter(self.datamodel.obj.name.in_(crd_names)).all()
                        # for db_crd in db_crds:
                        #     db_crd.status = 'Deleted'
                        # db.session.commit()
                        item.status = 'Deleted'
                        item.change_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        db.session.commit()
                        push_message(conf.get('ADMIN_USER', '').split(','), 'stop %s %s' % (crd_info['plural'],item.name))


                except Exception as e:
                    flash(str(e), "danger")

    def pre_delete(self, item):
        self.base_muldelete([item])

    @expose("/stop/<crd_id>")
    def stop(self, crd_id):
        crd = db.session.query(self.datamodel.obj).filter_by(id=crd_id).first()
        self.base_muldelete([crd])
        flash(__('清理完成'), 'success')
        self.update_redirect()
        return redirect(self.get_redirect())

    @action("stop_all", "停止", "停止所有选中的workflow?", "fa-trash", single=False)
    def stop_all(self, items):
        self.base_muldelete(items)
        self.update_redirect()
        return redirect(self.get_redirect())

    # @event_logger.log_this
    # @expose("/list/")
    # @has_access
    # def list(self):
    #     self.base_list()
    #     widgets = self._list()
    #     res = self.render_template(
    #         self.list_template, title=self.list_title, widgets=widgets
    #     )
    #     return res

    @action("muldelete", "删除", "确定删除所选记录?", "fa-trash", single=False)
    def muldelete(self, items):
        self.base_muldelete(items)
        for item in items:
            db.session.delete(item)

        db.session.commit()
        return json.dumps(
            {
                "success": [],
                "fail": []
            }, indent=4, ensure_ascii=False
        )


class Workflow_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query.order_by(self.model.create_time.desc())
        return query.filter(
            or_(
                self.model.labels.contains('"%s"' % g.user.username),
            )
        ).order_by(self.model.create_time.desc())


default_status_icon = '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"><svg t="1673257968006" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4593" xmlns:xlink="http://www.w3.org/1999/xlink" width="32" height="32"><path d="M512 51.2c254.08 0 460.8 206.72 460.8 460.8s-206.72 460.8-460.8 460.8S51.2 766.08 51.2 512 257.92 51.2 512 51.2M512 0C229.248 0 0 229.248 0 512s229.248 512 512 512 512-229.248 512-512S794.752 0 512 0L512 0z" fill="#ffffff" p-id="4594"></path><path d="M470.976 642.624C470.72 633.6 470.656 626.88 470.656 622.4c0-26.496 3.776-49.344 11.264-68.608 5.504-14.528 14.4-29.12 26.624-43.904 9.024-10.752 25.216-26.432 48.576-47.04s38.592-37.056 45.568-49.344 10.496-25.6 10.496-40.128c0-26.24-10.24-49.344-30.72-69.184S536.768 274.368 507.008 274.368c-28.736 0-52.736 9.024-72 27.008S403.136 347.52 397.12 385.728L327.744 377.472c6.272-51.264 24.832-90.496 55.68-117.76S455.104 218.88 505.856 218.88c53.76 0 96.64 14.656 128.64 43.904s48 64.64 48 106.112c0 24-5.632 46.144-16.896 66.368s-33.28 44.864-65.984 73.856C577.6 528.64 563.264 542.976 556.48 552.256S544.768 572.096 541.504 584.128s-5.12 31.488-5.632 58.496L470.976 642.624zM466.88 777.984l0-76.864 76.864 0 0 76.864L466.88 777.984z" fill="#ffffff" p-id="4595"></path></svg>'
status_icon = {
    "Running": '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"><svg t="1673257838593" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="8669" xmlns:xlink="http://www.w3.org/1999/xlink" width="32" height="32"><path d="M935.005091 459.752727a34.909091 34.909091 0 1 1 49.361454 49.361455l-78.382545 78.382545a34.816 34.816 0 0 1-49.338182 0l-78.405818-78.382545a34.909091 34.909091 0 1 1 49.361455-49.361455l14.801454 14.824728C818.525091 311.738182 678.330182 186.181818 508.928 186.181818c-130.466909 0-250.484364 76.706909-305.710545 195.397818a34.932364 34.932364 0 0 1-63.301819-29.463272C206.522182 208.896 351.418182 116.363636 508.904727 116.363636c210.152727 0 383.534545 159.953455 404.992 364.474182l21.085091-21.085091z m-73.960727 189.021091a34.932364 34.932364 0 0 1 16.965818 46.382546C811.310545 838.353455 666.461091 930.909091 508.951273 930.909091c-210.106182 0-383.534545-159.953455-404.968728-364.497455l-21.108363 21.108364a34.909091 34.909091 0 1 1-49.384727-49.361455l78.42909-78.42909a34.909091 34.909091 0 0 1 49.338182 0l78.382546 78.42909a34.909091 34.909091 0 1 1-49.338182 49.338182l-14.824727-14.801454C199.354182 735.534545 339.549091 861.090909 508.951273 861.090909c130.490182 0 250.507636-76.706909 305.710545-195.397818a34.909091 34.909091 0 0 1 46.382546-16.919273z" fill="#ffffff" p-id="8670"></path></svg>',
    "Error": '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"><svg t="1673257802788" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2718" xmlns:xlink="http://www.w3.org/1999/xlink" width="32" height="32"><path d="M512 128c211.2 0 384 172.8 384 384s-172.8 384-384 384-384-172.8-384-384 172.8-384 384-384m0-64C262.4 64 64 262.4 64 512s198.4 448 448 448 448-198.4 448-448-198.4-448-448-448z" fill="#ffffff" p-id="2719"></path><path d="M377.6 646.4m-32 0a32 32 0 1 0 64 0 32 32 0 1 0-64 0Z" fill="#ffffff" p-id="2720"></path><path d="M646.4 377.6m-32 0a32 32 0 1 0 64 0 32 32 0 1 0-64 0Z" fill="#ffffff" p-id="2721"></path><path d="M377.6 377.6m-32 0a32 32 0 1 0 64 0 32 32 0 1 0-64 0Z" fill="#ffffff" p-id="2722"></path><path d="M646.4 646.4m-32 0a32 32 0 1 0 64 0 32 32 0 1 0-64 0Z" fill="#ffffff" p-id="2723"></path><path d="M353.6 625.152l271.552-271.552 45.248 45.248-271.552 271.552z" fill="#ffffff" p-id="2724"></path><path d="M353.6 398.848l45.248-45.248 271.552 271.552-45.248 45.248z" fill="#ffffff" p-id="2725"></path></svg>',
    "Failed": '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"><svg t="1673257802788" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2718" xmlns:xlink="http://www.w3.org/1999/xlink" width="32" height="32"><path d="M512 128c211.2 0 384 172.8 384 384s-172.8 384-384 384-384-172.8-384-384 172.8-384 384-384m0-64C262.4 64 64 262.4 64 512s198.4 448 448 448 448-198.4 448-448-198.4-448-448-448z" fill="#ffffff" p-id="2719"></path><path d="M377.6 646.4m-32 0a32 32 0 1 0 64 0 32 32 0 1 0-64 0Z" fill="#ffffff" p-id="2720"></path><path d="M646.4 377.6m-32 0a32 32 0 1 0 64 0 32 32 0 1 0-64 0Z" fill="#ffffff" p-id="2721"></path><path d="M377.6 377.6m-32 0a32 32 0 1 0 64 0 32 32 0 1 0-64 0Z" fill="#ffffff" p-id="2722"></path><path d="M646.4 646.4m-32 0a32 32 0 1 0 64 0 32 32 0 1 0-64 0Z" fill="#ffffff" p-id="2723"></path><path d="M353.6 625.152l271.552-271.552 45.248 45.248-271.552 271.552z" fill="#ffffff" p-id="2724"></path><path d="M353.6 398.848l45.248-45.248 271.552 271.552-45.248 45.248z" fill="#ffffff" p-id="2725"></path></svg>',
    "Freeze": '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"><svg t="1673257894401" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="3575" xmlns:xlink="http://www.w3.org/1999/xlink" width="32" height="32"><path d="M499.599575 66.187445c-246.554372 0-447.133643 200.579271-447.133643 447.119317 0 246.559489 200.579271 447.134667 447.133643 447.134667 246.545162 0 447.12034-200.575178 447.12034-447.134667C946.719916 266.765693 746.144738 66.187445 499.599575 66.187445L499.599575 66.187445zM499.599575 929.249009c-229.351572 0-415.937131-186.590676-415.937131-415.94327 0-229.337245 186.585559-415.922804 415.937131-415.922804 229.338269 0 415.928944 186.585559 415.928944 415.922804C915.52852 742.659357 728.937844 929.249009 499.599575 929.249009L499.599575 929.249009zM818.027905 457.640951M369.890398 553.076534c-6.92369 10.175757-15.363919 23.098068-25.329898 38.758746-9.976212 15.668865-20.247136 32.348756-30.823006 50.048883-10.586103 17.701151-21.057595 35.30611-31.43392 52.79646-10.376325 17.500583-19.2269 33.169447-26.550703 46.998407-5.702885-6.924713-11.500938-12.512988-17.395182-16.784268-5.903453-4.273327-12.722766-8.030907-20.446681-11.292184 9.765411-10.986215 20.141736-24.204262 31.127951-39.673581 10.986215-15.459087 21.763676-32.04381 32.348756-49.743938 10.576893-17.701151 20.847817-35.801391 30.824029-54.32221 9.965979-18.511609 19.016099-36.317137 27.160593-53.407374 4.472871 8.954953 9.155521 16.175401 14.037714 21.667485C358.294292 543.616068 363.7874 548.604686 369.890398 553.076534zM333.87923 425.511183c-17.909905-21.973454-35.916001-41.704844-54.017264-59.204403-18.110473-17.49035-35.706223-32.749892-52.79646-45.777603l26.245757-31.127951c19.531845 16.279778 38.653345 33.474392 57.373709 51.575656 18.711154 18.110473 36.412304 35.916001 53.101405 53.40635L333.87923 425.511183zM510.274706 614.113679c-5.703909 2.850931-12.311397 8.144494-19.836791 15.869433-7.534604 7.735172-18.215874 20.552082-32.04381 38.452777-4.482081 5.702885-9.975189 12.20702-16.479323 19.531845-6.513344 7.323802-13.533224 14.858406-21.057595 22.582321-7.534604 7.735172-15.163351 15.46932-22.88829 23.194258-7.735172 7.724938-15.259542 14.647605-22.583344 20.751626-5.29254-5.702885-10.480702-10.376325-15.563464-14.038738-5.092995-3.66139-12.111852-7.53358-21.057595-11.596106 26.855648-16.269545 51.365878-36.107359 73.54911-59.510372 22.172999-23.393803 43.230594-52.386114 63.172785-86.976934 10.986215 9.766434 22.373567 15.870456 34.17945 18.312064 4.882194 0.819668 7.525394 2.651386 7.934716 5.493108C517.999644 609.029894 515.557012 611.67207 510.274706 614.113679zM477.314013 357.150236c-22.793122 0-43.945885 0.410346-63.47773 1.220804-19.531845 0.819668-37.031405 2.040473-52.490491 3.662413l0-40.894386c17.900695 2.041496 37.737486 3.566222 59.510372 4.578272 21.763676 1.020236 44.861744 1.52575 69.275783 1.52575 3.252067-7.724938 6.102998-15.048741 8.544607-21.972431 2.442632-6.91448 4.77884-14.038738 7.018857-21.36254 2.232854-7.323802 4.369517-14.953573 6.40999-22.88829 2.030239-7.934716 4.063549-16.984836 6.104021-27.160593 8.544607 4.482081 17.289782 7.934716 26.243711 10.376325 8.945743 2.441609 17.702174 4.072759 26.245757 4.883217 4.063549 0.410346 6.303566 2.345418 6.713912 5.799076 0.402159 3.461845-1.63115 6.208399-6.101975 8.239662-4.072759 2.040473-9.261944 7.629771-15.565511 16.784268-6.313799 9.155521-14.552437 24.929786-24.718985 47.303352l87.892793 0c28.886911 0 54.321186-0.505513 76.29464-1.52575 21.973454-1.011027 42.515303-2.536776 61.646012-4.578272l0 41.504276c-18.310018-2.031263-38.556131-3.452635-60.731176-4.272303-22.183232-0.810459-47.913243-1.220804-77.209476-1.220804l-100.709703 0c-7.735172 17.909905-15.059997 33.980929-21.973454 48.219211-6.92369 14.248516-13.637602 27.676339-20.141736 40.283472-6.513344 12.617365-13.122879 24.720008-19.836791 36.317137-6.713912 11.597129-13.732769 23.298636-21.057595 35.095309l123.903961 0c0-26.855648-0.409322-50.049906-1.220804-69.581752-0.819668-19.531845-2.240017-35.601846-4.273327-48.218188 8.945743 1.630127 17.492396 2.850931 25.63689 3.662413 8.134261 0.819668 16.268522 1.020236 24.413016 0.60989 6.503111-0.400113 10.07138 1.220804 10.68127 4.883217 0.611937 3.662413-1.935072 6.713912-7.628748 9.155521-3.2623 1.220804-5.798053 4.377704-7.628748 9.460466-1.830695 5.092995-2.7486 12.512988-2.7486 22.278399l0 67.750033c33.361829 0.410346 61.742203 0.209778 85.146239-0.610914 23.393803-0.810459 41.809221-2.231831 55.238069-4.272303l0 37.842887c-11.80793-0.811482-29.202089-1.52575-52.18657-2.136663-22.994714-0.610914-52.396347-0.915859-88.197738-0.915859L596.334757 723.367989c0 11.796674-1.019213 21.457708-3.049452 28.992311-2.042519 7.52437-6.313799 13.732769-12.818957 18.614963-6.513344 4.882194-15.773242 8.639774-27.771507 11.292184-12.005428 2.641153-27.981285 4.777817-47.913243 6.408967-0.409322-17.500583-7.322779-34.390251-20.751626-50.66082 17.090237 3.25309 30.517037 5.28333 40.284495 6.104021 9.765411 0.810459 17.289782 0.401136 22.582321-1.219781 5.284353-1.63115 8.639774-4.682649 10.07138-9.155521 1.421372-4.472871 2.136663-10.576893 2.136663-18.311041L559.104831 546.973536c-30.92943 0-56.258305 0.105401-75.990718 0.304945-19.741623 0.209778-35.706223 0.715291-47.913243 1.52575-12.20702 0.820692-21.467941 1.936095-27.771507 3.356444-6.313799 1.430582-11.292184 3.1569-14.953573 5.188162-0.410346-6.91448-1.735527-14.038738-3.967358-21.36254-2.241041-7.324826-4.988618-13.628392-8.239662-18.921955 5.692652-0.810459 10.881838-2.746554 15.563464-5.799076 4.673439-3.051499 10.376325-9.860579 17.090237-20.446681 6.713912-10.57587 15.154141-26.340925 25.329898-47.303352C448.417893 422.564062 461.445604 393.773342 477.314013 357.150236zM750.145865 746.561223c-8.144494-12.20702-16.994046-24.614607-26.549679-37.231973-9.565866-12.607132-19.532869-24.815175-29.90817-36.622082-10.377348-11.796674-20.751626-22.88829-31.128975-33.264615s-20.246113-19.426445-29.602202-27.160593l27.465538-25.635867c9.356089 8.54563 19.627013 18.520819 30.823006 29.90817 11.186783 11.396561 22.278399 23.193235 33.265638 35.401278 10.986215 12.20702 21.361517 24.214495 31.128975 36.011168 9.765411 11.805884 18.310018 22.382776 25.634844 31.738865L750.145865 746.561223z" fill="#ffffff" p-id="3576"></path></svg>',
    'Succeeded': '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"><svg t="1673257747369" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2739" xmlns:xlink="http://www.w3.org/1999/xlink" width="32" height="32"><path d="M508.248559 953.897386c-60.0824 0-118.381178-11.772115-173.275415-34.991956-53.009308-22.420639-100.612489-54.513568-141.484361-95.385441-40.872896-40.872896-72.964802-88.475054-95.386464-141.485384-23.218818-54.894238-34.990932-113.193015-34.990933-173.275416s11.773138-118.381178 34.990933-173.275415c22.421662-53.009308 54.513568-100.612489 95.386464-141.484362 40.871873-40.872896 88.475054-72.964802 141.484361-95.386464 54.895261-23.218818 113.194038-34.990932 173.275415-34.990932 60.0824 0 118.380154 11.773138 173.275416 34.990932 53.009308 22.421662 100.611465 54.513568 141.484361 95.386464 40.871873 40.871873 72.964802 88.475054 95.385441 141.484362 23.218818 54.895261 34.991955 113.194038 34.991955 173.275415s-11.773138 118.381178-34.991955 173.275416c-22.420639 53.010331-54.513568 100.612489-95.385441 141.485384-40.872896 40.871873-88.475054 72.964802-141.484361 95.385441-54.895261 23.218818-113.193015 34.991955-173.275416 34.991956z m0-839.844794c-217.641879 0-394.706597 177.064718-394.706596 394.706597 0 217.642902 177.064718 394.706597 394.706596 394.706597s394.705574-177.063695 394.705574-394.706597c0.001023-217.641879-177.063695-394.706597-394.705574-394.706597z" fill="#ffffff" p-id="2740"></path><path d="M448.493617 738.906893a25.485441 25.485441 0 0 1-16.624632-6.148024L250.937193 577.889663c-10.733459-9.188266-11.987009-25.337061-2.799766-36.07052 9.188266-10.732435 25.337061-11.988032 36.071543-2.798743l161.571863 138.297786L718.296483 361.414353c9.228175-10.69969 25.384134-11.890818 36.081777-2.660596 10.697643 9.229199 11.888771 25.38311 2.660596 36.081777L467.87606 730.034828c-5.058203 5.863545-12.199856 8.872065-19.382443 8.872065z" fill="#ffffff" p-id="2741"></path></svg>'
}
default_status_color = "#1C1C1C"
status_color = {
    "Running": "#2acc61",
    "Error": "#CB1B45",
    "Failed": "#CB1B45",
    "Freeze": "#33A6B8",
    "Succeeded": "#1B813E"
}


# http://data.tme.woa.com/frontend/commonRelation?backurl=/workflow_modelview/api/web/dag/idc/pipeline/crontab-standalone-train-znvv7
# list正在运行的workflow
class Workflow_ModelView_Base(Crd_ModelView_Base):
    base_filters = [["id", Workflow_Filter, lambda: []]]

    # 删除之前的 workflow和相关容器
    # @pysnooper.snoop()
    def delete_more(self, workflow):
        try:
            k8s_client = py_k8s.K8s(workflow.pipeline.project.cluster.get('KUBECONFIG', ''))
            k8s_client.delete_workflow(
                all_crd_info=conf.get("CRD_INFO", {}),
                namespace=workflow.namespace,
                run_id=json.loads(workflow.labels).get("run-id", '')
            )
            workflow.status = 'Deleted'
            workflow.change_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            db.session.commit()
        except Exception as e:
            print(e)

    @event_logger.log_this
    @expose("/stop/<crd_id>")
    def stop(self, crd_id):
        workflow = db.session.query(self.datamodel.obj).filter_by(id=crd_id).first()
        self.delete_more(workflow)

        flash(__('清理完成'), 'success')
        url = conf.get('MODEL_URLS', {}).get('workflow', '')
        return redirect(url)

    label_title = _('运行实例')
    datamodel = SQLAInterface(Workflow)
    list_columns = ['project', 'pipeline_url', 'cluster', 'create_time', 'change_time', 'elapsed_time', 'final_status', 'status', 'username', 'log', 'stop']
    search_columns = ['status', 'labels', 'name', 'cluster', 'annotations', 'spec', 'status_more', 'username', 'create_time']
    cols_width = {
        "project": {"type": "ellip2", "width": 100},
        "pipeline_url": {"type": "ellip2", "width": 300},
        "create_time": {"type": "ellip2", "width": 200},
        "change_time": {"type": "ellip2", "width": 200},
        "final_status": {"type": "ellip1", "width": 250},
        "elapsed_time":{"type": "ellip1", "width": 150},
    }
    spec_label_columns = {
        "final_status": _('删除前状态')
    }
    show_columns = ['name', 'namespace', 'create_time', 'status', 'task_status', 'annotations_html', 'labels_html', 'spec_html', 'status_more_html', 'info_json_html']
    crd_name = 'workflow'

    def get_dag(self, cluster_name, namespace, workflow_name, node_name=''):

        k8s_client = py_k8s.K8s(conf.get('CLUSTERS', {}).get(cluster_name, {}).get('KUBECONFIG', ''))
        crd_info = conf.get('CRD_INFO', {}).get('workflow', {})
        try_num=3
        workflow_obj=None
        # 尝试三次查询
        while not workflow_obj and try_num>0:
            workflow_obj = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'],
                                                  plural=crd_info['plural'], namespace=namespace, name=workflow_name)
            workflow_model = db.session.query(Workflow).filter_by(name=workflow_name).first()
            if not workflow_obj:
                if workflow_model:
                    workflow_obj = workflow_model.to_json()
            try_num-=1
            if not workflow_obj:
                time.sleep(2)
        # 没有查询到就返回空
        if not workflow_obj:
            return {}, {}, {}, None

        # print(workflow_obj)
        labels = json.loads(workflow_obj.get('labels', "{}"))
        spec = json.loads(workflow_obj.get('spec', '{}'))
        nodes_spec = {}
        for node in spec['templates']:
            if node['name'] != spec['entrypoint']:
                nodes_spec[node['name']] = node

        annotations = json.loads(workflow_obj.get('annotations', '{}'))
        status_more = json.loads(workflow_obj.get('status_more', '{}'))

        layout_config = {}
        dag_config = []
        self.node_detail_config = {}

        layout_config["create_time"] = workflow_obj['create_time']
        layout_config['search'] = ''
        layout_config["status"] = workflow_obj['status']
        layout_config.update(labels)
        layout_config['progress'] = status_more.get('progress', '0/0')
        layout_config["start_time"] = k8s_client.to_local_time(status_more.get('startedAt',''))
        layout_config['finish_time'] = k8s_client.to_local_time(status_more.get('finishedAt',''))

        layout_config['crd_json'] = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "metadata": {
                "annotations": core.decode_unicode_escape(annotations),
                "name": workflow_name,
                "labels": labels,
                "namespace": namespace
            },
            "spec": spec,
            "status": status_more
        }


        if int(layout_config.get("pipeline-id", '0')):
            pipeline = db.session.query(Pipeline).filter_by(id=int(layout_config.get("pipeline-id", '0'))).first()
            if pipeline:
                layout_config['pipeline-name'] = pipeline.name
                layout_config['pipeline-describe'] = pipeline.describe

        dag_default_status_icon = '<svg t="1673492959659" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7570" width="200" height="200"><path d="M512 51.2c254.08 0 460.8 206.72 460.8 460.8s-206.72 460.8-460.8 460.8S51.2 766.08 51.2 512 257.92 51.2 512 51.2M512 0C229.248 0 0 229.248 0 512s229.248 512 512 512 512-229.248 512-512S794.752 0 512 0L512 0z" fill="#D1D3D4" p-id="7571"></path><path d="M470.976 642.624C470.72 633.6 470.656 626.88 470.656 622.4c0-26.496 3.776-49.344 11.264-68.608 5.504-14.528 14.4-29.12 26.624-43.904 9.024-10.752 25.216-26.432 48.576-47.04s38.592-37.056 45.568-49.344 10.496-25.6 10.496-40.128c0-26.24-10.24-49.344-30.72-69.184S536.768 274.368 507.008 274.368c-28.736 0-52.736 9.024-72 27.008S403.136 347.52 397.12 385.728L327.744 377.472c6.272-51.264 24.832-90.496 55.68-117.76S455.104 218.88 505.856 218.88c53.76 0 96.64 14.656 128.64 43.904s48 64.64 48 106.112c0 24-5.632 46.144-16.896 66.368s-33.28 44.864-65.984 73.856C577.6 528.64 563.264 542.976 556.48 552.256S544.768 572.096 541.504 584.128s-5.12 31.488-5.632 58.496L470.976 642.624zM466.88 777.984l0-76.864 76.864 0 0 76.864L466.88 777.984z" fill="#D1D3D4" p-id="7572"></path></svg>'
        dag_status_icon = {
            "Running": '<svg t="1673492809370" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4015" width="200" height="200"><path d="M512 1024c-282.304 0-512-229.696-512-512s229.696-512 512-512 512 229.696 512 512S794.304 1024 512 1024zM512 48.768C256.576 48.768 48.768 256.576 48.768 512S256.576 975.168 512 975.168 975.168 767.424 975.168 512 767.424 48.768 512 48.768z" fill="#45c872" p-id="4016"></path><path d="M806.336 478.912c0 0-423.36-238.976-427.328-239.616C339.52 232.64 328.256 261.632 327.04 289.536L326.016 290.368c0 0 2.496 465.216 3.904 468.928 11.648 31.296 45.76 31.296 70.144 17.664l0.64 0.256c0 0 392.64-226.944 395.84-229.504C817.344 530.624 836.48 505.664 806.336 478.912z" fill="#45c872" p-id="4017"></path></svg>',
            "Error": '<svg t="1673492920895" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="6459" width="200" height="200"><path d="M549.044706 512l166.189176-166.249412a26.383059 26.383059 0 0 0 0-36.98447 26.383059 26.383059 0 0 0-37.044706 0L512 475.015529l-166.249412-166.249411a26.383059 26.383059 0 0 0-36.98447 0 26.383059 26.383059 0 0 0 0 37.044706L475.015529 512l-166.249411 166.249412a26.383059 26.383059 0 0 0 0 36.98447 26.383059 26.383059 0 0 0 37.044706 0L512 548.984471l166.249412 166.249411a26.383059 26.383059 0 0 0 36.98447 0 26.383059 26.383059 0 0 0 0-37.044706L548.984471 512zM512 1024a512 512 0 1 1 0-1024 512 512 0 0 1 0 1024z" fill="#E84335" p-id="6460"></path></svg>',
            "Failed": '<svg t="1673492920895" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="6459" width="200" height="200"><path d="M549.044706 512l166.189176-166.249412a26.383059 26.383059 0 0 0 0-36.98447 26.383059 26.383059 0 0 0-37.044706 0L512 475.015529l-166.249412-166.249411a26.383059 26.383059 0 0 0-36.98447 0 26.383059 26.383059 0 0 0 0 37.044706L475.015529 512l-166.249411 166.249412a26.383059 26.383059 0 0 0 0 36.98447 26.383059 26.383059 0 0 0 37.044706 0L512 548.984471l166.249412 166.249411a26.383059 26.383059 0 0 0 36.98447 0 26.383059 26.383059 0 0 0 0-37.044706L548.984471 512zM512 1024a512 512 0 1 1 0-1024 512 512 0 0 1 0 1024z" fill="#E84335" p-id="6460"></path></svg>',
            'Succeeded': '<svg t="1673492905079" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="5495" width="200" height="200"><path d="M512 512m-512 0a512 512 0 1 0 1024 0 512 512 0 1 0-1024 0Z" fill="#52C41A" p-id="5496"></path><path d="M178.614857 557.860571a42.496 42.496 0 0 1 60.123429-60.050285l85.942857 87.625143a42.496 42.496 0 0 1-60.050286 60.123428L178.614857 557.860571z m561.005714-250.148571a42.496 42.496 0 1 1 65.097143 54.637714L394.459429 725.577143a42.496 42.496 0 0 1-65.097143-54.637714l410.112-363.373715z" fill="#FFFFFF" p-id="5497"></path></svg>'
        }

        layout_config['icon'] = dag_status_icon.get(workflow_obj['status'], dag_default_status_icon)
        layout_config['title'] = workflow_name
        layout_config['right_button'] = []

        if workflow_model:
            layout_config["right_button"].append(
                {
                    "label": "终止",
                    "url": f"/workflow_modelview/api/stop/{workflow_model.id}"
                }
            )
        layout_config['right_button'].append(
            {
                "label": __("任务流"),
                "url": f'/pipeline_modelview/api/web/{pipeline.id}'
            }
        )
        layout_config['detail'] = [
            [
                {
                    "name": "cluster",
                    "label": __("集群"),
                    "value": cluster_name
                },
                # {
                #     "name": "describe",
                #     "label": "描述",
                #     "value": pipeline.describe
                # },
                {
                    "name": "id",
                    "label": "id",
                    "value": f'{pipeline.id}({pipeline.describe})'
                },
                {
                    "name": "status",
                    "label": __("状态"),
                    "value": workflow_obj['status']
                },
                {
                    "name": "message",
                    "label": __("消息"),
                    "value": status_more.get('message', '')
                },

            ],
            [
                {
                    "name": "create_time",
                    "label": __("创建时间"),
                    "value": workflow_obj['create_time']
                },
                {
                    "name": "start_time",
                    "label": __("开始时间"),
                    "value": k8s_client.to_local_time(status_more['startedAt']) if 'startedAt' in status_more else ''
                },
                {
                    "name": "finish_time",
                    "label": __("结束时间"),
                    "value": k8s_client.to_local_time(status_more['finishedAt']) if 'finishedAt' in status_more else ''
                },
                {
                    "name": "run_id",
                    "label": "run-id",
                    "value": labels.get("run-id", '')
                }
            ],
            [
                {
                    "name": "created_by",
                    "label": __("创建人"),
                    "value": pipeline.created_by.username
                },
                {
                    "name": "run_user",
                    "label": __("执行人"),
                    "value": labels.get("run-rtx", '')
                },

                {
                    "name": "progress",
                    "label": __("进度"),
                    "value": status_more.get('progress', '0/0')
                },
                {
                    "name": "schedule_type",
                    "label": __("调度类型"),
                    "value": labels.get("schedule_type", 'once')
                },

            ],
        ]

        templates = {}
        for node in spec['templates']:
            templates[node['name']] = node

        # @pysnooper.snoop()
        def fill_child(self, dag, upstream_node_name):
            try:
                childs = status_more['nodes'][upstream_node_name].get('children', [])
                for child in childs:
                    try:
                        # pod_name = child   # 这里不对，这里是workflow 名后随机生成
                        status = status_more['nodes'][child].get('phase', 'unknown')
                        task_name = status_more['nodes'][child]['templateName']
                        pod_name = workflow_name + "-" + task_name + "-" + child.replace(workflow_name, '').strip('-')
                        s3_key = ''
                        metric_key = ''
                        output_key = ''
                        artifacts = status_more['nodes'][child].get('outputs', {}).get('artifacts', [])
                        for artifact in artifacts:
                            if artifact['name'] == 'main-logs':
                                s3_key = artifact.get('s3', {}).get('key', '')
                            if artifact['name'] == 'metric':
                                metric_key = artifact.get('s3', {}).get('key', '')
                            if artifact['name'] == 'output':
                                output_key = artifact.get('s3', {}).get('key', '')
                        retry = nodes_spec[task_name].get('retryStrategy', {}).get("limit", 0)

                        # 对于可重试节点的发起节点，没有日志和执行命令，
                        displayName = status_more['nodes'][child].get('displayName', '')
                        displayName = displayName.replace("(0)", '(first)')
                        match = re.findall("(\([1-9]+\))", displayName)
                        if len(match) > 0:
                            retry_index = match[0].replace("(", '').replace(")", '')
                            displayName = displayName.replace(match[0], __('(第%s次重试)')%retry_index)
                        title = nodes_spec[task_name].get('metadata', {}).get("annotations", {}).get("task", pod_name)+f"({displayName})"
                        node_type = status_more['nodes'][child]['type']
                        if node_type == "Retry":
                            title += __("(有%s次重试机会)")%retry
                        if node_type == 'Skipped':
                            title += "(skip)"

                        nodeSelector = nodes_spec[task_name].get('nodeSelector', {})
                        node_selector = ''
                        for key in nodeSelector:
                            node_selector += key + "=" + nodeSelector[key] + ","
                        node_selector = node_selector.strip(',')
                        requests_resource = nodes_spec[task_name].get('container', {}).get("resources", {}).get("requests", {})
                        resource_gpu = "0"
                        for resource_name in list(conf.get('GPU_RESOURCE',{}).values()):
                            if resource_name in requests_resource:
                                resource_gpu = str(requests_resource.get(resource_name,"0"))
                                break

                        ui_node = {
                            "node_type": node_type,
                            "nid": status_more['nodes'][child]['id'],
                            "pid": status_more['nodes'][upstream_node_name]['id'],
                            "title": title,
                            "pod": pod_name,
                            "start_time": k8s_client.to_local_time(status_more['nodes'][child].get('startedAt','')),
                            "finish_time": k8s_client.to_local_time(status_more['nodes'][child].get('finishedAt','')),
                            "detail_url": self.route_base + f"/web/node_detail/{cluster_name}/{namespace}/{workflow_name}/{child}",
                            "name": pod_name,
                            "outputs": status_more['nodes'][child].get('outputs', {}),
                            # "icon": '<svg t="1671087371964" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4320" width="200" height="200"><path d="M109.714286 73.142857c-21.942857 0-36.571429 14.628571-36.571429 36.571429v804.571428c0 21.942857 14.628571 36.571429 36.571429 36.571429h804.571428c21.942857 0 36.571429-14.628571 36.571429-36.571429v-804.571428c0-21.942857-14.628571-36.571429-36.571429-36.571429h-804.571428z m0-73.142857h804.571428c58.514286 0 109.714286 51.2 109.714286 109.714286v804.571428c0 58.514286-51.2 109.714286-109.714286 109.714286h-804.571428C51.2 1024 0 972.8 0 914.285714v-804.571428C0 51.2 51.2 0 109.714286 0z m438.857143 292.571429h219.428571c21.942857 0 36.571429 14.628571 36.571429 36.571428s-14.628571 36.571429-36.571429 36.571429h-219.428571c-21.942857 0-36.571429-14.628571-36.571429-36.571429s14.628571-36.571429 36.571429-36.571428z m-219.428572 438.857142c21.942857 0 36.571429-14.628571 36.571429-36.571428S351.085714 658.285714 329.142857 658.285714s-36.571429 14.628571-36.571428 36.571429 14.628571 36.571429 36.571428 36.571428z m0 73.142858C270.628571 804.571429 219.428571 753.371429 219.428571 694.857143S270.628571 585.142857 329.142857 585.142857 438.857143 636.342857 438.857143 694.857143 387.657143 804.571429 329.142857 804.571429z m-7.314286-446.171429L241.371429 277.942857c-14.628571-14.628571-36.571429-14.628571-51.2 0-14.628571 14.628571-14.628571 36.571429 0 51.2L292.571429 431.542857c7.314286 7.314286 21.942857 14.628571 29.257142 14.628572s21.942857 0 29.257143-7.314286l153.6-153.6c14.628571-14.628571 14.628571-36.571429 0-51.2-14.628571-14.628571-36.571429-14.628571-51.2 0L321.828571 358.4zM548.571429 658.285714h219.428571c21.942857 0 36.571429 14.628571 36.571429 36.571429s-14.628571 36.571429-36.571429 36.571428h-219.428571c-21.942857 0-36.571429-14.628571-36.571429-36.571428s14.628571-36.571429 36.571429-36.571429z" p-id="4321"></path></svg>',
                            "icon": status_icon.get(status, default_status_icon),
                            "status": {
                                "label": status,
                                "icon": status_icon.get(status, default_status_icon)
                            },
                            "message": status_more['nodes'][child].get('message', ''),
                            "node_shape": "rectangle",
                            "color": status_color.get(status, default_status_color),
                            "task_name": task_name,
                            "task_id": nodes_spec[task_name].get('metadata', {}).get("labels", {}).get("task-id", ''),
                            "task_label": nodes_spec[task_name].get('metadata', {}).get("annotations", {}).get("task", ''),
                            "volumeMounts": nodes_spec[task_name].get('container', {}).get("volumeMounts", []),
                            "volumes": nodes_spec[task_name].get('volumes', []),
                            "node_selector": node_selector,
                            "s3_key": s3_key,
                            "metric_key": metric_key,
                            "output_key":output_key,
                            "retry": retry,
                            "resource_cpu": str(nodes_spec[task_name].get('container', {}).get("resources", {}).get("requests", {}).get("cpu", '0')),
                            "resource_memory": str(nodes_spec[task_name].get('container', {}).get("resources", {}).get("requests", {}).get("memory", '0')),
                            "resource_gpu": resource_gpu,
                            "children": []
                        }
                        if node_name == child and not self.node_detail_config:
                            self.node_detail_config = ui_node

                        fill_child(self, ui_node['children'], child)
                        dag.append(ui_node)
                    except Exception as e:
                        print(e)
            except Exception as e:
                print(e)

        fill_child(self, dag_config, workflow_name)
        return layout_config, dag_config, self.node_detail_config, workflow_obj

    @expose("/web/log/<cluster_name>/<namespace>/<workflow_name>/<pod_name>", methods=["GET", ])
    @expose("/web/log_node/<cluster_name>/<namespace>/<workflow_name>/<pod_name>", methods=["GET", ])
    @expose("/web/log/<cluster_name>/<namespace>/<workflow_name>/<pod_name>/<file_name>", methods=["GET", ])
    def log_node(self, cluster_name, namespace, workflow_name, pod_name,file_name='main.log'):
        log = self.get_minio_content(f'{workflow_name}/{pod_name}/{file_name}')
        if '/web/log/' in request.path:
            from wtforms.widgets.core import HTMLString, html_params

            return Markup("<pre><code>%s</code></pre>"%log)
        return jsonify({
            "status": 0,
            "message": "",
            "result": {
                "type": "html",
                "value": Markup(log)
            }
        })

    # @pysnooper.snoop(watch_explode=())
    def get_minio_content(self, key, decompress=True, download=False):
        if request.host=='127.0.0.1':
            return
        content = ''
        from minio import Minio
        try:
            minioClient = Minio(
                endpoint=conf.get('MINIO_HOST','minio.kubeflow:9000'),  # minio.kubeflow:9000    '9.135.92.226:9944'
                access_key='minio',
                secret_key='minio123',
                secure=False
            )
            if download:
                save_path = "/mlpipeline/" + key
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                minioClient.fget_object('mlpipeline', key, save_path)
                response = make_response(send_from_directory(os.path.dirname(save_path), os.path.basename(save_path), as_attachment=True,conditional=True))
                return response
            response = None
            try:
                response = minioClient.get_object('mlpipeline', key)
                content = response.data
            except Exception as e:
                content = str(e)
                print(e)
                return content
            finally:
                if response:
                    response.close()
                    response.release_conn()

        except Exception as e:
            print(e)
            return str(e)

        if decompress:
            if '.zip' in key:
                import zlib
                content = zlib.decompress(content)
            if '.tgz' in key:
                path = 'minio/' + key
                if os.path.exists(path):
                    os.remove(path)

                os.makedirs(os.path.dirname(path), exist_ok=True)
                file = open(path,mode='wb')
                file.write(content)
                file.close()
                import tarfile
                # 打开tgz文件
                with tarfile.open(path, 'r:gz') as tar:
                    # 解压所有文件到指定目录
                    tar.extractall(os.path.dirname(path))

                files = os.listdir(os.path.dirname(path))
                content = ''
                for file in files:
                    path = os.path.join(os.path.dirname(path),file)
                    if os.path.isfile(path) and path[path.rindex('.'):] in ['.txt','.json','.log','.csv']:
                        content = ''.join(open(os.path.join(os.path.dirname(path),file)).readlines())
                        content += '\n'

        # print(key[key.rindex('.'):])
        if key[key.rindex('.'):] in ['.txt','.json','.log','.csv']:
            if type(content)==bytes:
                content = content.decode()

            # 删除 ANSI 转义序列
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            content = ansi_escape.sub('', content)
        return content

    @expose("/web/node_detail/<cluster_name>/<namespace>/<workflow_name>/<node_name>", methods=["GET", ])
    def web_node_detail(self,cluster_name,namespace,workflow_name,node_name):
        layout_config, dag_config,node_detail_config,workflow = self.get_dag(cluster_name, namespace, workflow_name,node_name)
        # print(node_detail_config)
        if not node_detail_config:
            return jsonify({})
        del node_detail_config['children']
        pod_name = node_detail_config["pod"]
        # 获取pod信息
        k8s_client = py_k8s.K8s(conf.get('CLUSTERS', {}).get(cluster_name, {}).get('KUBECONFIG', ''))
        pod_yaml = __('pod未发现')
        pod_status = ''
        pod = None
        try:
            # from kubernetes.client import V1Pod
            pod = k8s_client.get_pod_humanized(namespace=namespace, pod_name=pod_name)
            if pod:
                pod_status = pod.get("status", {}).get('phase', 'Unknown')
                # 去除一些比好看的，没必要的信息
                for key in copy.deepcopy(pod['metadata'].get('annotations',{})):
                    if 'cni' in key or 'kubectl' in key:
                        del pod['metadata']['annotations'][key]

                for key in ['creationTimestamp','resourceVersion','uid']:
                    if key in pod['metadata']:
                        del pod['metadata'][key]

                for key in ['initContainers','enableServiceLinks','dnsPolicy','tolerations','terminationGracePeriodSeconds']:
                    if key in pod['spec']:
                        del pod['spec'][key]

                volumes = []
                for index,volume in enumerate(pod['spec'].get('volumes',[])):
                    if 'my-minio-cred' in str(volume) or 'kube-api-access' in str(volume):
                        pass
                    else:
                        volumes.append(volume)

                pod['spec']['volumes']=volumes

                if 'status' in pod:
                    del pod['status']

                containers = []
                for container in pod['spec']['containers']:
                    if 'emissary' in str(container['command']):
                        container_temp = copy.deepcopy(container)
                        envs = []
                        for env in container_temp.get('env',[]):
                            if 'ARGO_' not in str(env):
                                envs.append(env)

                        container_temp['env']=envs

                        containers.append(container_temp)

                pod['spec']['containers'] = containers
                pod_yaml = json.dumps(pod, indent=4, ensure_ascii=False, default=str)

                # import yaml
                # pod_yaml = yaml.safe_dump(yaml.load(pod_yaml, Loader=yaml.SafeLoader), default_flow_style=False, indent=4)
                # # print(pod)

        except Exception as e:
            print(e)

        host_url = "http://" + conf.get("CLUSTERS", {}).get(cluster_name, {}).get("HOST", request.host)

        online_pod_log_url = "/k8s/web/log/%s/%s/%s/main" % (cluster_name, namespace, pod_name)
        offline_pod_log_url = f'/workflow_modelview/api/web/log/{cluster_name}/{namespace}/{workflow_name}/{pod_name}/main.log'
        offline_pod_metric_url = f'/workflow_modelview/api/web/log/{cluster_name}/{namespace}/{workflow_name}/{pod_name}/metric.tgz'
        debug_online_url = "/k8s/web/debug/%s/%s/%s/main" % (cluster_name, namespace, pod_name)
        grafana_pod_url = host_url+conf.get('GRAFANA_TASK_PATH','/grafana/d/pod-info/pod-info?var-pod=')+pod_name
        labels = json.loads(workflow.get('labels', "{}"))
        pipeline_name = labels.get('pipeline-name', workflow_name)
        bind_pod_url = f'/k8s/web/search/{cluster_name}/{namespace}/{pipeline_name}'

        echart_option = ''
        metric_content = ''
        try:
            if node_detail_config['metric_key']:
                metric_content = self.get_minio_content(node_detail_config['metric_key'],decompress=True)
                # print(metric_content)
                metric_content = metric_content
        except Exception as e:
            print(e)

        message = node_detail_config.get('message', '')
        node_type = node_detail_config.get('node_type', '')
        volumes = {}
        for vol in node_detail_config['volumes']:
            if vol.get("persistentVolumeClaim", {}).get("claimName", ''):
                volumes[vol['name']] = vol.get("persistentVolumeClaim", {}).get("claimName", '') + "(pvc)"
            if vol.get("hostPath", {}).get("path", ''):
                volumes[vol['name']] = vol.get("hostPath", {}).get("path", '') + "(hostpath)"
            if vol.get("emptyDir", {}).get("medium", ''):
                volumes[vol['name']] = vol.get("emptyDir", {}).get("medium", '')

        tab1 = [
            {
                "tabName": __("输入输出"),
                "content": [
                    {
                        "groupName": __("消息"),
                        "groupContent": {
                            "label": __('消息'),
                            "value": message if message else __('运行正常'),
                            "type": 'html'
                        }
                    },
                    {
                        "groupName": __("任务详情"),
                        "groupContent": {
                            "label": __('任务详情'),
                            "value": {
                                "task id": node_detail_config['task_id'],
                                "task name": node_detail_config['task_name'],
                                "task label": node_detail_config['task_label'],
                                "task start": node_detail_config['start_time'],
                                "task finish": node_detail_config['finish_time'],
                                "pod name": node_detail_config['pod'],
                                "log path": node_detail_config['s3_key'],
                                "metric path": node_detail_config['metric_key'],
                                "retry": node_detail_config['retry'],
                                "node selector": node_detail_config['node_selector'],
                                "resource cpu": node_detail_config['resource_cpu'],
                                "resource memory": node_detail_config['resource_memory'],
                                "resource gpu": node_detail_config['resource_gpu']
                            },
                            "type": 'map'
                        }
                    },
                    {
                        "groupName": __("挂载详情"),
                        "groupContent": {
                            "label": __('挂载详情'),
                            "value": dict(
                                [[__('容器路径'), __('主机路径')]] + [[item['mountPath'], volumes.get(item['name'], '')] for item in node_detail_config['volumeMounts']]),
                            "type": 'map'
                        }
                    },
                ],
                "bottomButton": []
            }
        ]
        if node_type == 'Pod':
            tab1[0]['bottomButton'] = [
                {
                    # "icon": '<svg t="1672911530794" class="icon" viewBox="0 0 1025 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2682" width="200" height="200"><path d="M331.502964 331.496564c-11.110365 11.110365-11.110365 29.107109 0 40.191874l321.816594 321.790994c11.110365 11.161565 29.107109 11.110365 40.191874 0.0256 11.110365-11.110365 11.110365-29.081509 0-40.191874L371.720439 331.496564C360.610073 320.411799 342.61333 320.411799 331.502964 331.496564z" p-id="2683"></path><path d="M96.2141 59.958213 59.990213 96.2077c-79.97415 79.97415-79.99975 209.637745 0 289.611895l126.719604 126.719604c62.668604 62.719804 155.749913 75.878163 231.602476 40.243074 0.281599-0.128 0.537598-0.230399 0.844797-0.332799 0.537598-0.255999 1.177596-0.486398 1.715195-0.742398-0.0512-0.128 0.0512 0.128 0 0 2.713592-1.356796 5.171184-3.07199 7.449577-5.350383 11.238365-11.238365 11.238365-29.491108 0-40.755073-9.036772-9.011172-22.24633-10.598367-33.049497-5.171184-0.0512-0.1536 0.0768 0.1536 0 0-56.575823 25.72792-125.849207 15.180753-172.364261-31.359902L103.433277 349.621308c-59.980613-60.006212-59.980613-157.234709 0-217.215321L132.386787 103.426877c59.980613-59.980613 157.234709-59.955013 217.215321 0l119.474827 119.474827c47.283052 47.257452 57.292621 117.810832 30.131106 174.847454 0 0 0.0256-0.0256 0 0-0.0768 0.204799 0.1024-0.204799 0 0 0.0512 0.0256-0.0256-0.0256 0 0-3.839988 10.239968-2.227193 22.707129 6.015981 30.950303 11.238365 11.238365 29.491108 11.263965 40.755073 0 2.303993-2.303993 4.377586-4.915185 5.759982-7.679976 0.1536 0.0256-0.179199-0.0256 0 0 37.196684-76.313361 24.217524-170.905066-39.167878-234.316068l-126.719604-126.719604C305.851844-19.990337 176.16265-19.990337 96.2141 59.958213z" p-id="2684"></path><path d="M963.411389 927.155503l-36.249487 36.223887c-79.97415 79.97415-209.637745 79.97415-289.611895-0.0256l-126.719604-126.694004c-62.668604-62.668604-75.878163-155.775513-40.217474-231.602476 0.128-0.281599 0.230399-0.537598 0.332799-0.844797 0.255999-0.537598 0.511998-1.203196 0.742398-1.715195 0.128 0.0512-0.128-0.0512 0 0 1.356796-2.713592 3.07199-5.171184 5.350383-7.449577 11.238365-11.238365 29.491108-11.238365 40.780673 0 8.985572 9.011172 10.572767 22.220731 5.119984 33.049497 0.179199 0.0768-0.128-0.0768 0 0-25.72792 56.601423-15.155153 125.823607 31.385502 172.364261l119.474827 119.449227c60.006212 60.006212 157.234709 60.006212 217.215321 0l28.95351-28.95351c60.006212-60.006212 59.980613-157.234709 0-217.189721l-119.449227-119.474827c-47.283052-47.257452-117.810832-57.292621-174.847454-30.131106-0.0256 0 0.0256 0 0 0-0.204799 0.128 0.230399-0.0768 0 0-0.0256 0 0.0256 0.0256 0 0-10.239968 3.865588-22.707129 2.252793-30.950303-6.015981-11.238365-11.263965-11.238365-29.491108-0.0256-40.755073 2.303993-2.278393 4.889585-4.351986 7.705576-5.734382-0.0256-0.128 0.0256 0.179199 0 0 76.313361-37.222284 170.905066-24.217524 234.290468 39.167878l126.719604 126.719604C1043.359939 717.492158 1043.359939 847.181353 963.411389 927.155503z" p-id="2685"></path></svg>',
                    "text": __("在线日志") + ("" if pod else '(no)'),
                    "url": online_pod_log_url
                },
                {
                    # "icon": '<svg t="1672911530794" class="icon" viewBox="0 0 1025 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2682" width="200" height="200"><path d="M331.502964 331.496564c-11.110365 11.110365-11.110365 29.107109 0 40.191874l321.816594 321.790994c11.110365 11.161565 29.107109 11.110365 40.191874 0.0256 11.110365-11.110365 11.110365-29.081509 0-40.191874L371.720439 331.496564C360.610073 320.411799 342.61333 320.411799 331.502964 331.496564z" p-id="2683"></path><path d="M96.2141 59.958213 59.990213 96.2077c-79.97415 79.97415-79.99975 209.637745 0 289.611895l126.719604 126.719604c62.668604 62.719804 155.749913 75.878163 231.602476 40.243074 0.281599-0.128 0.537598-0.230399 0.844797-0.332799 0.537598-0.255999 1.177596-0.486398 1.715195-0.742398-0.0512-0.128 0.0512 0.128 0 0 2.713592-1.356796 5.171184-3.07199 7.449577-5.350383 11.238365-11.238365 11.238365-29.491108 0-40.755073-9.036772-9.011172-22.24633-10.598367-33.049497-5.171184-0.0512-0.1536 0.0768 0.1536 0 0-56.575823 25.72792-125.849207 15.180753-172.364261-31.359902L103.433277 349.621308c-59.980613-60.006212-59.980613-157.234709 0-217.215321L132.386787 103.426877c59.980613-59.980613 157.234709-59.955013 217.215321 0l119.474827 119.474827c47.283052 47.257452 57.292621 117.810832 30.131106 174.847454 0 0 0.0256-0.0256 0 0-0.0768 0.204799 0.1024-0.204799 0 0 0.0512 0.0256-0.0256-0.0256 0 0-3.839988 10.239968-2.227193 22.707129 6.015981 30.950303 11.238365 11.238365 29.491108 11.263965 40.755073 0 2.303993-2.303993 4.377586-4.915185 5.759982-7.679976 0.1536 0.0256-0.179199-0.0256 0 0 37.196684-76.313361 24.217524-170.905066-39.167878-234.316068l-126.719604-126.719604C305.851844-19.990337 176.16265-19.990337 96.2141 59.958213z" p-id="2684"></path><path d="M963.411389 927.155503l-36.249487 36.223887c-79.97415 79.97415-209.637745 79.97415-289.611895-0.0256l-126.719604-126.694004c-62.668604-62.668604-75.878163-155.775513-40.217474-231.602476 0.128-0.281599 0.230399-0.537598 0.332799-0.844797 0.255999-0.537598 0.511998-1.203196 0.742398-1.715195 0.128 0.0512-0.128-0.0512 0 0 1.356796-2.713592 3.07199-5.171184 5.350383-7.449577 11.238365-11.238365 29.491108-11.238365 40.780673 0 8.985572 9.011172 10.572767 22.220731 5.119984 33.049497 0.179199 0.0768-0.128-0.0768 0 0-25.72792 56.601423-15.155153 125.823607 31.385502 172.364261l119.474827 119.449227c60.006212 60.006212 157.234709 60.006212 217.215321 0l28.95351-28.95351c60.006212-60.006212 59.980613-157.234709 0-217.189721l-119.449227-119.474827c-47.283052-47.257452-117.810832-57.292621-174.847454-30.131106-0.0256 0 0.0256 0 0 0-0.204799 0.128 0.230399-0.0768 0 0-0.0256 0 0.0256 0.0256 0 0-10.239968 3.865588-22.707129 2.252793-30.950303-6.015981-11.238365-11.263965-11.238365-29.491108-0.0256-40.755073 2.303993-2.278393 4.889585-4.351986 7.705576-5.734382-0.0256-0.128 0.0256 0.179199 0 0 76.313361-37.222284 170.905066-24.217524 234.290468 39.167878l126.719604 126.719604C1043.359939 717.492158 1043.359939 847.181353 963.411389 927.155503z" p-id="2685"></path></svg>',
                    "text": __("在线调试") + ("" if pod_status == 'Running' else '(no)'),
                    "url": debug_online_url
                },
                {
                    # "icon": '<svg t="1672911530794" class="icon" viewBox="0 0 1025 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2682" width="200" height="200"><path d="M331.502964 331.496564c-11.110365 11.110365-11.110365 29.107109 0 40.191874l321.816594 321.790994c11.110365 11.161565 29.107109 11.110365 40.191874 0.0256 11.110365-11.110365 11.110365-29.081509 0-40.191874L371.720439 331.496564C360.610073 320.411799 342.61333 320.411799 331.502964 331.496564z" p-id="2683"></path><path d="M96.2141 59.958213 59.990213 96.2077c-79.97415 79.97415-79.99975 209.637745 0 289.611895l126.719604 126.719604c62.668604 62.719804 155.749913 75.878163 231.602476 40.243074 0.281599-0.128 0.537598-0.230399 0.844797-0.332799 0.537598-0.255999 1.177596-0.486398 1.715195-0.742398-0.0512-0.128 0.0512 0.128 0 0 2.713592-1.356796 5.171184-3.07199 7.449577-5.350383 11.238365-11.238365 11.238365-29.491108 0-40.755073-9.036772-9.011172-22.24633-10.598367-33.049497-5.171184-0.0512-0.1536 0.0768 0.1536 0 0-56.575823 25.72792-125.849207 15.180753-172.364261-31.359902L103.433277 349.621308c-59.980613-60.006212-59.980613-157.234709 0-217.215321L132.386787 103.426877c59.980613-59.980613 157.234709-59.955013 217.215321 0l119.474827 119.474827c47.283052 47.257452 57.292621 117.810832 30.131106 174.847454 0 0 0.0256-0.0256 0 0-0.0768 0.204799 0.1024-0.204799 0 0 0.0512 0.0256-0.0256-0.0256 0 0-3.839988 10.239968-2.227193 22.707129 6.015981 30.950303 11.238365 11.238365 29.491108 11.263965 40.755073 0 2.303993-2.303993 4.377586-4.915185 5.759982-7.679976 0.1536 0.0256-0.179199-0.0256 0 0 37.196684-76.313361 24.217524-170.905066-39.167878-234.316068l-126.719604-126.719604C305.851844-19.990337 176.16265-19.990337 96.2141 59.958213z" p-id="2684"></path><path d="M963.411389 927.155503l-36.249487 36.223887c-79.97415 79.97415-209.637745 79.97415-289.611895-0.0256l-126.719604-126.694004c-62.668604-62.668604-75.878163-155.775513-40.217474-231.602476 0.128-0.281599 0.230399-0.537598 0.332799-0.844797 0.255999-0.537598 0.511998-1.203196 0.742398-1.715195 0.128 0.0512-0.128-0.0512 0 0 1.356796-2.713592 3.07199-5.171184 5.350383-7.449577 11.238365-11.238365 29.491108-11.238365 40.780673 0 8.985572 9.011172 10.572767 22.220731 5.119984 33.049497 0.179199 0.0768-0.128-0.0768 0 0-25.72792 56.601423-15.155153 125.823607 31.385502 172.364261l119.474827 119.449227c60.006212 60.006212 157.234709 60.006212 217.215321 0l28.95351-28.95351c60.006212-60.006212 59.980613-157.234709 0-217.189721l-119.449227-119.474827c-47.283052-47.257452-117.810832-57.292621-174.847454-30.131106-0.0256 0 0.0256 0 0 0-0.204799 0.128 0.230399-0.0768 0 0-0.0256 0 0.0256 0.0256 0 0-10.239968 3.865588-22.707129 2.252793-30.950303-6.015981-11.238365-11.263965-11.238365-29.491108-0.0256-40.755073 2.303993-2.278393 4.889585-4.351986 7.705576-5.734382-0.0256-0.128 0.0256 0.179199 0 0 76.313361-37.222284 170.905066-24.217524 234.290468 39.167878l126.719604 126.719604C1043.359939 717.492158 1043.359939 847.181353 963.411389 927.155503z" p-id="2685"></path></svg>',
                    "text": __("相关容器") + ("" if pod else '(no)'),
                    "url": bind_pod_url
                },
                {
                    # "icon": '<svg t="1672911530794" class="icon" viewBox="0 0 1025 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2682" width="200" height="200"><path d="M331.502964 331.496564c-11.110365 11.110365-11.110365 29.107109 0 40.191874l321.816594 321.790994c11.110365 11.161565 29.107109 11.110365 40.191874 0.0256 11.110365-11.110365 11.110365-29.081509 0-40.191874L371.720439 331.496564C360.610073 320.411799 342.61333 320.411799 331.502964 331.496564z" p-id="2683"></path><path d="M96.2141 59.958213 59.990213 96.2077c-79.97415 79.97415-79.99975 209.637745 0 289.611895l126.719604 126.719604c62.668604 62.719804 155.749913 75.878163 231.602476 40.243074 0.281599-0.128 0.537598-0.230399 0.844797-0.332799 0.537598-0.255999 1.177596-0.486398 1.715195-0.742398-0.0512-0.128 0.0512 0.128 0 0 2.713592-1.356796 5.171184-3.07199 7.449577-5.350383 11.238365-11.238365 11.238365-29.491108 0-40.755073-9.036772-9.011172-22.24633-10.598367-33.049497-5.171184-0.0512-0.1536 0.0768 0.1536 0 0-56.575823 25.72792-125.849207 15.180753-172.364261-31.359902L103.433277 349.621308c-59.980613-60.006212-59.980613-157.234709 0-217.215321L132.386787 103.426877c59.980613-59.980613 157.234709-59.955013 217.215321 0l119.474827 119.474827c47.283052 47.257452 57.292621 117.810832 30.131106 174.847454 0 0 0.0256-0.0256 0 0-0.0768 0.204799 0.1024-0.204799 0 0 0.0512 0.0256-0.0256-0.0256 0 0-3.839988 10.239968-2.227193 22.707129 6.015981 30.950303 11.238365 11.238365 29.491108 11.263965 40.755073 0 2.303993-2.303993 4.377586-4.915185 5.759982-7.679976 0.1536 0.0256-0.179199-0.0256 0 0 37.196684-76.313361 24.217524-170.905066-39.167878-234.316068l-126.719604-126.719604C305.851844-19.990337 176.16265-19.990337 96.2141 59.958213z" p-id="2684"></path><path d="M963.411389 927.155503l-36.249487 36.223887c-79.97415 79.97415-209.637745 79.97415-289.611895-0.0256l-126.719604-126.694004c-62.668604-62.668604-75.878163-155.775513-40.217474-231.602476 0.128-0.281599 0.230399-0.537598 0.332799-0.844797 0.255999-0.537598 0.511998-1.203196 0.742398-1.715195 0.128 0.0512-0.128-0.0512 0 0 1.356796-2.713592 3.07199-5.171184 5.350383-7.449577 11.238365-11.238365 29.491108-11.238365 40.780673 0 8.985572 9.011172 10.572767 22.220731 5.119984 33.049497 0.179199 0.0768-0.128-0.0768 0 0-25.72792 56.601423-15.155153 125.823607 31.385502 172.364261l119.474827 119.449227c60.006212 60.006212 157.234709 60.006212 217.215321 0l28.95351-28.95351c60.006212-60.006212 59.980613-157.234709 0-217.189721l-119.449227-119.474827c-47.283052-47.257452-117.810832-57.292621-174.847454-30.131106-0.0256 0 0.0256 0 0 0-0.204799 0.128 0.230399-0.0768 0 0-0.0256 0 0.0256 0.0256 0 0-10.239968 3.865588-22.707129 2.252793-30.950303-6.015981-11.238365-11.263965-11.238365-29.491108-0.0256-40.755073 2.303993-2.278393 4.889585-4.351986 7.705576-5.734382-0.0256-0.128 0.0256 0.179199 0 0 76.313361-37.222284 170.905066-24.217524 234.290468 39.167878l126.719604 126.719604C1043.359939 717.492158 1043.359939 847.181353 963.411389 927.155503z" p-id="2685"></path></svg>',
                    "text": __("资源使用"),
                    "url": grafana_pod_url
                },
                {
                    # "icon": '<svg t="1672911530794" class="icon" viewBox="0 0 1025 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2682" width="200" height="200"><path d="M331.502964 331.496564c-11.110365 11.110365-11.110365 29.107109 0 40.191874l321.816594 321.790994c11.110365 11.161565 29.107109 11.110365 40.191874 0.0256 11.110365-11.110365 11.110365-29.081509 0-40.191874L371.720439 331.496564C360.610073 320.411799 342.61333 320.411799 331.502964 331.496564z" p-id="2683"></path><path d="M96.2141 59.958213 59.990213 96.2077c-79.97415 79.97415-79.99975 209.637745 0 289.611895l126.719604 126.719604c62.668604 62.719804 155.749913 75.878163 231.602476 40.243074 0.281599-0.128 0.537598-0.230399 0.844797-0.332799 0.537598-0.255999 1.177596-0.486398 1.715195-0.742398-0.0512-0.128 0.0512 0.128 0 0 2.713592-1.356796 5.171184-3.07199 7.449577-5.350383 11.238365-11.238365 11.238365-29.491108 0-40.755073-9.036772-9.011172-22.24633-10.598367-33.049497-5.171184-0.0512-0.1536 0.0768 0.1536 0 0-56.575823 25.72792-125.849207 15.180753-172.364261-31.359902L103.433277 349.621308c-59.980613-60.006212-59.980613-157.234709 0-217.215321L132.386787 103.426877c59.980613-59.980613 157.234709-59.955013 217.215321 0l119.474827 119.474827c47.283052 47.257452 57.292621 117.810832 30.131106 174.847454 0 0 0.0256-0.0256 0 0-0.0768 0.204799 0.1024-0.204799 0 0 0.0512 0.0256-0.0256-0.0256 0 0-3.839988 10.239968-2.227193 22.707129 6.015981 30.950303 11.238365 11.238365 29.491108 11.263965 40.755073 0 2.303993-2.303993 4.377586-4.915185 5.759982-7.679976 0.1536 0.0256-0.179199-0.0256 0 0 37.196684-76.313361 24.217524-170.905066-39.167878-234.316068l-126.719604-126.719604C305.851844-19.990337 176.16265-19.990337 96.2141 59.958213z" p-id="2684"></path><path d="M963.411389 927.155503l-36.249487 36.223887c-79.97415 79.97415-209.637745 79.97415-289.611895-0.0256l-126.719604-126.694004c-62.668604-62.668604-75.878163-155.775513-40.217474-231.602476 0.128-0.281599 0.230399-0.537598 0.332799-0.844797 0.255999-0.537598 0.511998-1.203196 0.742398-1.715195 0.128 0.0512-0.128-0.0512 0 0 1.356796-2.713592 3.07199-5.171184 5.350383-7.449577 11.238365-11.238365 29.491108-11.238365 40.780673 0 8.985572 9.011172 10.572767 22.220731 5.119984 33.049497 0.179199 0.0768-0.128-0.0768 0 0-25.72792 56.601423-15.155153 125.823607 31.385502 172.364261l119.474827 119.449227c60.006212 60.006212 157.234709 60.006212 217.215321 0l28.95351-28.95351c60.006212-60.006212 59.980613-157.234709 0-217.189721l-119.449227-119.474827c-47.283052-47.257452-117.810832-57.292621-174.847454-30.131106-0.0256 0 0.0256 0 0 0-0.204799 0.128 0.230399-0.0768 0 0-0.0256 0 0.0256 0.0256 0 0-10.239968 3.865588-22.707129 2.252793-30.950303-6.015981-11.238365-11.263965-11.238365-29.491108-0.0256-40.755073 2.303993-2.278393 4.889585-4.351986 7.705576-5.734382-0.0256-0.128 0.0256 0.179199 0 0 76.313361-37.222284 170.905066-24.217524 234.290468 39.167878l126.719604 126.719604C1043.359939 717.492158 1043.359939 847.181353 963.411389 927.155503z" p-id="2685"></path></svg>',
                    "text": __("离线日志") + ("" if node_type == 'Pod' and pod_status != 'Running' and pod_status!='Pending' else '(no)'),
                    "url": offline_pod_log_url
                }
            ]
        tab2 = [
            {
                "tabName": __("pod信息"),
                "content": [
                    {
                        "groupName": __("yaml信息"),
                        "groupContent": {
                            "value": Markup(pod_yaml),
                            "type": 'text'
                        }
                    }
                ],
                "bottomButton": []
            }
        ]

        tab3 = [
            {
                "tabName": __("在线日志"),
                "content": [
                    {
                        "groupName": __("在线日志"),
                        "groupContent": {
                            "value": f'/k8s/web/log/{cluster_name}/{namespace}/{pod_name}/main' if pod_status == 'Running' else __("pod未发现"),
                            "type": 'iframe' if pod_status == 'Running' else "html"
                        }
                    }
                ],
                "bottomButton": []
            }
        ]

        tab4 = [
            {
                "tabName": __("在线调试"),
                "content": [
                    {
                        "groupName": __("在线调试"),
                        "groupContent": {
                            "value": f'/k8s/web/exec/{cluster_name}/{namespace}/{pod_name}/main' if pod_status == 'Running' else __("pod已停止运行"),
                            "type": 'iframe' if pod_status == 'Running' else 'html'
                        }
                    }
                ],
                "bottomButton": []
            }
        ]

        tab5 = [
            {
                "tabName": __("相关容器"),
                "content": [
                    {
                        "groupName": __("相关容器"),
                        "groupContent": {
                            "value": {
                                "url": host_url+conf.get('K8S_DASHBOARD_CLUSTER','/k8s/dashboard/cluster/')+f"#/search?namespace={namespace}&q={pod_name}",
                                "target": "div.kd-chrome-container.kd-bg-background",
                            } if pod_status else __("pod未发现"),
                            "type": 'iframe' if pod_status else "html"
                        }
                    }
                ],
                "bottomButton": []
            }
        ]

        tab6 = [
            {
                "tabName": __("资源使用情况"),
                "content": [
                    {
                        "groupName": __("资源使用情况"),
                        "groupContent": {
                            "value": {
                                "url": host_url + conf.get('GRAFANA_TASK_PATH', '/grafana/d/pod-info/pod-info?var-pod=') + pod_name,
                            },
                            "type": 'iframe'
                        }
                    }
                ],
                "bottomButton": []
            }
        ]
        tab7 = [
            {
                "tabName": __("结果可视化"),
                "content": [
                    {
                        "groupName": "",
                        "groupContent": {
                            "value": Markup("提示：仅企业版支持任务结果、模型指标、数据集可视化预览"),
                            # options的值
                            "type": 'html'
                        }
                    },

                ],
                "bottomButton": []
            },
        ]
        if not metric_content:
            echart_demos_file = os.listdir('myapp/utils/echart/')
            for file in echart_demos_file:
                # print(file)
                file_path = os.path.join('myapp/utils/echart/',file)
                can = ['area-stack.json', 'rose.json', 'mix-line-bar.json', 'pie-nest.json', 'bar-stack.json',
                       'candlestick-simple.json', 'graph-simple.json', 'tree-polyline.json', 'sankey-simple.json',
                       'radar.json', 'sunburst-visualMap.json', 'parallel-aqi.json', 'funnel.json',
                       'sunburst-visualMap.json', 'scatter-effect.json']
                not_can = ['bar3d-punch-card.json', 'simple-surface.json']# 不行的。

                # if '.json' in file and file in []:
                if '.json' in file and file in can:
                    echart_option = ''.join(open(file_path).readlines())
                    # print(echart_option)
                    tab7[0]['content'].append(
                        {
                            "groupName": __("任务结果示例：")+file.replace('.json','')+__("类型图表"),
                            "groupContent": {
                                "value": echart_option,  # options的值
                                "type": 'echart'
                            }
                        }
                    )

        tab8 = [
            {
                "tabName": __("workflow信息"),
                "content": [
                    {
                        "groupName": __("json信息"),
                        "groupContent": {
                            "value": Markup(json.dumps(layout_config.get("crd_json",{}),indent=4,ensure_ascii=False)),
                            "type": 'text'
                        }
                    }
                ],
                "bottomButton": []
            }
        ]

        node_detail = {
            "detail": tab1 + tab2 + tab7 + tab8,
            "control": {
                "width": "700px"
            }
        }
        return jsonify({
            "status": 0,
            "message": "success",
            "result": node_detail
        }
        )

    @expose("/web/dag/<cluster_name>/<namespace>/<workflow_name>", methods=["GET", ])
    # @pysnooper.snoop()
    def web_dag(self, cluster_name, namespace, workflow_name):
        layout_config, dag_config, node_detail_config, workflow = self.get_dag(cluster_name, namespace, workflow_name)
        back = {
            "control": {
                "node_ops": ["detail", "explore"],  # 节点可进行的操作  详情查看/节点上下链路探索，以后才有功能再加
                "direction": "vertical",  # 或者 vertical  horizontal,
            },
            "dag": dag_config,
            "layout": layout_config
        }
        return jsonify(
            {
                "status": 0,
                "message": "success",
                "result": back
            }
        )

    @expose("/web/layout/<cluster_name>/<namespace>/<workflow_name>", methods=["GET", ])
    def web_layout(self, cluster_name, namespace, workflow_name):
        layout_config, dag_config, node_detail_config, workflow = self.get_dag(cluster_name, namespace, workflow_name)
        layout_config['title'] = f"{cluster_name} {namespace} {workflow_name} {layout_config['start_time']} {layout_config['finish_time']}"
        return jsonify(
            {
                "status": 0,
                "message": "success",
                "result": layout_config
            }
        )

class Workflow_ModelView(Workflow_ModelView_Base, MyappModelView, DeleteMixin):
    datamodel = SQLAInterface(Workflow)


appbuilder.add_view_no_menu(Workflow_ModelView)


# 添加api
class Workflow_ModelView_Api(Workflow_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Workflow)
    route_base = '/workflow_modelview/api'


appbuilder.add_api(Workflow_ModelView_Api)

