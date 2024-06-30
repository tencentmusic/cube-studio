import base64
import uuid
import random
import re
import shutil
import logging
from myapp.models.model_chat import Chat, ChatLog
import requests
import time
from myapp.forms import MySelect2Widget, MyBS3TextFieldWidget
import multiprocessing
from flask import Flask, render_template, send_file
import pandas as pd
from myapp.exceptions import MyappException
from sqlalchemy.exc import InvalidRequestError
import datetime
from flask import Response,flash,g
from flask_appbuilder import action
from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from wtforms.validators import DataRequired, Regexp
from myapp import app, appbuilder
from wtforms import StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MySelect2Widget
from flask import jsonify, Markup, make_response, stream_with_context
from .baseApi import MyappModelRestApi
from flask import g, request, redirect
import urllib
import json, os, sys
import emoji,re
from werkzeug.utils import secure_filename
import pysnooper
from sqlalchemy import or_
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app, appbuilder, db
from flask_appbuilder import expose
import threading
import queue
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)
from myapp import cache
conf = app.config
logging.getLogger("sseclient").setLevel(logging.INFO)
max_len=2000

class Chat_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query
        return query.filter(
            or_(
                self.model.owner.contains(g.user.username),
                self.model.owner.contains('*')
            )
        )


default_icon = '<svg t="1708691376697" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4274" width="50" height="50"><path d="M512 127.0272c-133.4144 0-192.4864 72.1792-192.4864 192.4864V656.384h384.9856V319.5136c-0.0128-120.3072-78.72-192.4864-192.4992-192.4864z" fill="#A67C52" p-id="4275"></path><path d="M560.128 487.9488h-96.256l-24.0512 96.2432v312.8064h144.3584V584.192z" fill="#DBB59A" p-id="4276"></path><path d="M223.2576 704.4992c-45.2352 45.2352-48.128 192.4864-48.128 192.4864H512L415.7568 608.256s-169.8944 73.6256-192.4992 96.2432z m577.4848 0C778.112 681.8688 608.256 608.256 608.256 608.256L512 896.9984h336.8576s-2.88-147.264-48.1152-192.4992z" fill="#48A0DC" p-id="4277"></path><path d="M584.1792 584.192L512 896.9984h24.064l72.1792-168.4352 120.3072-24.064-144.3712-120.3072z m-288.7296 120.3072l120.3072 24.064 72.1792 168.4352H512L439.8208 584.192l-144.3712 120.3072z" fill="#FFFFFF" p-id="4278"></path><path d="M578.2144 270.976c-18.8288 41.6384-83.7888 72.6016-162.4576 72.6016h-47.7824c1.0496 47.1424 5.44 83.7248 23.7312 120.3072 24.064 48.128 73.1648 96.2432 120.3072 96.2432S608.256 512 632.32 463.8848c21.4272-42.8416 23.7696-85.696 24.0256-145.5232-1.4208-27.0464-52.48-47.3856-78.1312-47.3856z" fill="#F6CBAD" p-id="4279"></path><path d="M723.6864 283.8912c-21.0176-75.904-107.8016-132.8-211.6864-132.8s-190.6688 56.896-211.6864 132.8c-16.0512 8.8064-28.928 21.504-28.928 35.6224v48.128c0 26.5728 45.6064 48.128 72.1792 48.128v-96.2432c0-66.4448 75.4048-120.3072 168.4352-120.3072s168.4352 53.8624 168.4352 120.3072v48.128c0 51.5712-32.448 95.552-78.0288 112.6656-6.4384-9.8816-17.5872-16.4224-30.2464-16.4224h-48.128c-19.9296 0-36.096 16.1664-36.096 36.096 0 19.9296 16.1664 36.096 36.096 36.096H572.16c3.52 0 6.912-0.512 10.1248-1.4464 70.9888-9.3312 128.0512-62.8608 142.6432-132.0576 15.4752-8.768 27.6992-21.1712 27.6992-34.9184v-48.128c-0.0128-14.144-12.8896-26.8416-28.9408-35.648z" fill="#4D4D4D" p-id="4280"></path></svg>'
icon_choices=[
    '<svg t="1682394317506" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2833" width="50" height="50"><path d="M431.207059 2.199998C335.414129 13.19899 257.420186 72.593947 219.024215 163.78688l-6.199996 14.797989-19.997985 5.799996C104.233299 210.582846 38.840347 279.776795 15.041364 372.369727c-6.999995 27.39698-8.999993 71.393948-4.199997 99.990927 7.399995 44.996967 26.597981 88.592935 53.795961 121.989911l9.198993 11.399991-5.199996 19.597986c-6.799995 26.597981-8.598994 74.593945-3.799997 103.190924 14.799989 87.392936 75.193945 163.58688 155.587886 196.383857 46.395966 18.998986 95.99193 24.797982 142.187895 16.798987l11.599992-1.999998 18.597986 17.598987c30.396978 28.596979 66.593951 48.395965 108.789921 59.994956 25.998981 6.999995 83.193939 8.999993 111.391918 3.599997 53.194961-9.799993 98.391928-33.797975 137.1889-72.794946 27.996979-28.196979 51.194963-64.393953 59.794956-93.591932 2.199998-6.999995 3.599997-8.599994 8.798993-9.799993 12.798991-2.598998 42.595969-13.39799 56.194959-20.196985 35.996974-17.998987 72.793947-49.195964 94.792931-80.593941 19.797985-28.197979 36.196973-65.993952 44.395967-102.990924 1.799999-7.799994 2.799998-24.997982 2.799998-48.995965 0-33.997975-0.6-38.796972-5.799996-58.995956-9.998993-38.795972-25.997981-71.993947-48.395964-100.190927l-10.198993-12.799991 4.399997-17.597987c26.79698-102.790925-16.798988-217.181841-105.391923-276.576797-30.996977-20.598985-58.194957-31.997977-95.59193-40.196971-22.397984-4.999996-70.993948-5.799996-91.991932-1.799998-12.399991 2.399998-12.99999 2.399998-15.799989-1.599999-4.598997-7.199995-34.795975-31.596977-52.794961-42.995969C548.196973 9.598993 486.603019-4.199997 431.207059 2.199998z m45.395967 67.793951c25.197982 2.399998 40.39697 6.399995 61.394955 16.198988 16.797988 7.799994 41.995969 23.397983 41.995969 25.997981 0 0.799999-45.595967 27.79798-101.390926 59.794956-55.995959 32.196976-104.591923 60.794955-108.19092 63.394954-14.799989 10.998992-14.399989 8.399994-14.59999 97.591928-0.2 43.995968-0.999999 110.389919-1.599998 147.387892l-1.199 67.393951-42.596968-24.397982-42.595969-24.397982 0.599999-134.988902c0.799999-154.386887 0.2-147.987892 19.597986-187.383862 29.797978-60.395956 86.792936-100.191927 151.987889-106.591922 8.199994-0.799999 15.398989-1.599999 15.998988-1.599999 0.6-0.2 9.798993 0.6 20.597985 1.599999z m268.977803 82.992939c73.393946 15.399989 132.189903 74.193946 147.387892 147.987892 3.599997 16.998988 4.599997 62.394954 1.599999 67.79495-1.199999 2.399998-22.797983-9.399993-108.590921-59.394957-105.391923-61.394955-107.191921-62.394954-117.989913-62.394954-10.799992 0-13.19999 1.399999-137.989899 73.593946l-126.989907 73.393946-0.599-49.395963c-0.2-27.19798 0.2-49.995963 1-50.795963 3.799997-3.599997 209.182847-121.189911 223.581836-127.989906 35.796974-16.797988 77.992943-21.397984 118.589913-12.798991z m-537.955606 362.369735c3.199998 4.599997 37.596972 25.398981 130.389904 78.993942 69.393949 39.796971 125.988908 72.993947 125.988908 73.593946 0 0.6-5.599996 4.199997-12.598991 8.199994-6.799995 3.799997-25.997981 14.797989-42.596968 24.397982l-30.196978 17.597987-107.790921-62.194954c-59.194957-34.196975-114.589916-67.393951-122.78991-73.793946-29.397978-22.597983-56.395959-63.793953-66.194952-101.190926-6.199995-24.197982-7.199995-60.794955-2.199998-84.992938 7.599994-36.996973 23.397983-66.994951 49.195964-93.792931 17.398987-17.997987 33.197976-29.396978 55.195959-40.195971l16.997988-8.199994 0.999999 127.589907 0.999999 127.589906 4.599997 6.398996zM750.379825 367.169731c56.394959 32.596976 108.389921 62.994954 115.589916 67.593951 43.396968 28.597979 73.593946 75.793944 81.99294 127.989906 3.599997 21.597984 1.599999 61.994955-3.999997 80.992941-8.998993 31.397977-24.996982 58.995957-47.594966 82.593939-17.598987 18.397987-48.195965 38.995971-65.794951 44.395967l-4.599997 1.399999v-124.189909c0-138.188899 0.4-133.389902-13.59899-143.387895-4.399997-2.999998-62.393954-37.196973-128.988906-75.593944-66.594951-38.596972-121.189911-70.393948-121.189911-70.993948-0.2-0.799999 83.592939-49.795964 85.192938-49.995964 0.4 0 46.595966 26.597981 102.991924 59.194957z m-181.385867 50.195963l54.99596 31.596977v127.989906l-55.19596 31.596977-55.194959 31.797977-39.196971-22.598983c-21.797984-12.398991-46.795966-26.99698-55.994959-32.196977l-16.398988-9.799993 0.399999-63.393953 0.6-63.394954 53.99496-31.396977c29.797978-17.198987 54.79596-31.397977 55.59596-31.397977 0.799999-0.2 26.197981 13.99999 56.394958 31.197977z m147.587892 85.592938l41.39697 23.797982v127.389907c0 139.787898-0.4 146.187893-11.999991 178.384869-11.597992 31.796977-36.595973 65.394952-64.593953 86.592937-6.799995 5.199996-21.397984 13.79899-32.396976 18.997986-51.995962 24.997982-109.59092 25.597981-162.586881 1.799999-12.598991-5.799996-40.39697-23.397983-40.396971-25.797982 0-0.6 46.996966-28.196979 104.191924-61.194955 57.394958-32.996976 107.190921-62.794954 110.789919-66.193951 3.799997-3.799997 7.399995-9.999993 8.799993-15.399989 1.599999-6.398995 2.199998-50.994963 2.199999-151.386889 0-78.392943 0.799999-141.987896 1.599999-141.587896 0.799999 0.2 20.197985 11.398992 42.995968 24.597982zM622.590919 732.139464c-3.799997 3.599997-205.38285 119.189913-221.781838 126.989907-26.597981 12.798991-47.995965 17.397987-79.792941 17.397987-19.798985 0-30.197978-0.999999-43.596968-4.199997-68.59395-16.997988-120.589912-66.193952-140.587897-133.787902-5.599996-18.798986-8.599994-57.395958-5.999996-75.193945l1.399999-9.199993 50.395963 29.197979c174.185872 100.391926 165.185879 95.59193 176.185871 95.591929 9.598993-0.2 16.597988-3.799997 137.1879-73.393946l126.989907-73.393946 0.599999 49.395964c0.2 26.99798-0.2 49.795964-0.999999 50.595963z" p-id="2834"></path></svg>',
    '<svg t="1686402272731" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2683" width="50" height="50"><path d="M0 895.984494l477.825318 128.015506V256.031012L0 128.015506zM887.425318 0l-384.542701 192.023259L119.456329 0v95.887583L503.378801 198.473652l384.542701-102.462023zM545.802544 256.031012v767.968988l478.197456-128.015506V128.015506z" fill="#006934" p-id="2684"></path></svg>',
    '<svg t="1686402383408" class="icon" viewBox="0 0 1068 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="9266" width="50" height="50"><path d="M553.2394 778.224h-0.05c-7.99 0-16.092-0.08-24.08-0.231-139.472-2.679-270.492-28.227-368.938-71.938C55.6134 659.625-1.2666 597.068 0.0224 529.92c1.252-65.252 57.605-124.407 158.69-166.561 94.734-39.506 220.97-61.264 355.443-61.264 7.972 0 16.073 0.077 24.073 0.226 139.475 2.673 270.495 28.221 368.947 71.943 104.553 46.43 161.437 108.98 160.144 176.124-1.248 65.262-57.613 124.412-158.693 166.567-94.715 39.508-220.944 61.269-355.386 61.269m-39.085-431.16c-128.673 0-248.76 20.528-338.137 57.802C93.6154 439.232 45.8614 485.124 44.9814 530.786c-0.904 47.198 47.734 96.106 133.445 134.166 93.081 41.336 217.934 65.515 351.543 68.075 7.702 0.152 15.512 0.224 23.224 0.224h0.046c128.648 0 248.713-20.528 338.087-57.796 82.392-34.364 130.153-80.26 131.03-125.926 0.906-47.205-47.732-96.111-133.44-134.162-93.09-41.346-217.94-65.518-351.546-68.08a1224.355 1224.355 0 0 0-23.214-0.223" p-id="9267"></path><path d="M756.9434 1005.515c-59.176 0-131.026-32.34-207.793-93.514-75.843-60.441-149.83-143.674-213.94-240.688-76.91-116.372-130.195-238.778-150.027-344.65-21.076-112.447-1.83-194.769 54.203-231.8C259.5284 81.556 283.4234 74.8 310.3924 74.8c59.177 0 131.035 32.332 207.792 93.514 75.848 60.446 149.832 143.676 213.941 240.694 76.904 116.374 130.182 238.765 150.027 344.642 21.07 112.456 1.82 194.771-54.209 231.803-20.144 13.311-44.028 20.062-71 20.062M310.3934 119.77c-18.002 0-33.549 4.246-46.209 12.614-39.39 26.032-52.073 93.829-34.802 186 18.758 100.116 69.668 216.65 143.35 328.13 61.543 93.136 132.243 172.772 204.446 230.318 67.73 53.982 131.58 83.713 179.766 83.713 18 0 33.55-4.247 46.205-12.606 39.388-26.034 52.08-93.831 34.798-186.008-18.754-100.11-69.662-216.642-143.339-328.127-61.548-93.135-132.243-172.777-204.45-230.319-67.738-53.98-131.576-83.715-179.766-83.715" p-id="9268"></path><path d="M336.2474 1022.571c-21.815 0-41.914-4.884-59.874-14.774-58.847-32.38-84.69-112.869-72.783-226.66 11.211-107.126 54.414-233.438 121.658-355.652C398.6414 292.103 493.0914 178.773 584.3814 114.555c10.16-7.147 24.19-4.707 31.334 5.448 7.145 10.163 4.707 24.185-5.456 31.332-85.885 60.418-175.404 168.245-245.604 295.836-64.417 117.082-105.735 237.347-116.342 338.652-9.757 93.266 8.37 159.819 49.734 182.574 41.363 22.758 107.278 2.454 180.845-55.713 79.893-63.172 159.371-162.448 223.786-279.529 70.2-127.587 113.367-260.914 118.428-365.804 0.605-12.404 11.138-21.978 23.546-21.378 12.402 0.6 21.976 11.14 21.38 23.544-5.38 111.497-50.557 251.94-123.949 385.319-67.24 122.217-150.808 226.318-235.3 293.128-62.346 49.294-120.885 74.607-170.536 74.607" p-id="9269"></path><path d="M655.3074 113.592c0 29.6 12.182 59.013 33.111 79.95 20.93 20.929 50.348 33.11 79.952 33.11 29.602 0 59.014-12.181 79.945-33.11 20.934-20.937 33.112-50.352 33.112-79.95 0-29.602-12.178-59.021-33.112-79.95C827.3854 12.717 797.9724 0.528 768.3714 0.528c-29.604 0-59.023 12.189-79.952 33.112-20.93 20.93-33.11 50.349-33.11 79.95" p-id="9270"></path></svg>',
    '<svg t="1686402633537" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="20717" width="50" height="50"><path d="M433.501009 539.493067c0 28.299636-22.999704 51.299341-51.299341 51.29934-28.299636 0-51.299341-22.999704-51.299341-51.29934 0-28.299636 22.999704-51.299341 51.299341-51.299341 28.399635 0 51.299341 22.999704 51.299341 51.299341z m208.297323-51.299341c-28.299636 0-51.299341 22.999704-51.299341 51.299341 0 28.299636 22.999704 51.299341 51.299341 51.29934 28.299636 0 51.299341-22.999704 51.299341-51.29934 0-28.299636-22.999704-51.299341-51.299341-51.299341zM860.895516 696.991043v-5.799926c-5.899924 17.09978-12.899834 33.299572-20.499736 48.99937 1.399982 9.699875 3.999949 23.099703 8.399892 38.599504 21.099729 32.599581 80.598964 74.199046 121.398439 84.898909-21.999717 9.29988-48.499377 13.19983-74.299045 9.199882 23.599697 29.699618 56.899269 56.699271 103.798666 72.699066-109.69859 98.898729-290.496267 101.098701-416.794643 24.299687-24.499685 7.199907-48.399378 10.999859-70.79909 10.999859s-46.399404-3.799951-70.79909-10.999859c-126.398376 76.799013-307.096053 74.599041-416.794644-24.299687 47.899384-16.399789 81.498953-44.099433 105.098649-74.499043-20.899731 0.79999-41.699464-2.999961-59.499235-10.599864 29.899616-7.7999 69.799103-32.299585 97.29875-57.599259 9.599877-25.499672 14.299816-48.199381 16.399789-62.699195-7.599902-15.799797-14.699811-31.999589-20.499737-48.99937v5.799926h-61.999203c-48.899372 0-88.598861-39.69949-88.598861-88.598862v-97.498747c0-33.699567 18.799758-62.99919 46.499402-77.998997 0.899988-138.398221 41.299469-246.496832 120.098457-321.195873C257.203275 37.599517 369.201835 0 512 0c142.798165 0 254.796725 37.599517 332.99572 111.698564 78.798987 74.69904 119.098469 182.797651 120.098457 321.195873 27.699644 14.999807 46.499402 44.299431 46.499402 77.998997v97.498747c0 48.899372-39.69949 88.598861-88.598861 88.598862h-62.099202zM823.99599 539.893062c0-22.799707-1.199985-44.199432-2.899962-64.999165-39.999486-36.59953-113.098547-64.199175-205.397361-73.89905 14.599812 13.099832 27.19965 30.599607 33.899565 55.599285-53.599311-39.199496-165.997867-51.299341-203.397386-93.3988-59.399237-39.299495-74.999036-95.69877-75.49903-72.399069-2.399969 111.798563-81.69895 198.997443-169.697819 211.297284-0.599992 12.299842-0.999987 24.699683-0.999987 37.699516 0 47.299392 7.699901 90.398838 20.799732 129.198339 49.399365 59.599234 131.098315 76.499017 203.397386 80.998959 13.399828-21.299726 43.499441-36.299533 78.59899-36.299533 47.49939 0 86.098893 27.399648 86.098894 61.099215s-38.499505 61.099215-86.098894 61.099215c-36.59953 0-67.69913-16.199792-80.198969-38.999499-50.499351-2.899963-105.998638-11.499852-155.498002-34.899552C336.702253 864.588889 444.600866 918.788192 512 918.788192c105.898639 0 311.99599-133.698282 311.99599-378.89513z m53.599312-117.598489h34.099561C906.094935 180.797676 768.196707 53.199316 512 53.199316S117.905065 180.797676 112.305137 422.294573h34.099561c12.499839-81.398954 38.799501-148.398093 78.798988-199.797432C289.002866 140.498194 385.501626 98.898729 512 98.898729c126.598373 0 223.097133 41.599465 286.796314 123.598412 39.999486 51.399339 66.399147 118.398478 78.798988 199.797432z" fill="#626264" p-id="20718"></path></svg>'
]
prompt_default= __('''
你是一个AI助手，以下```中的内容是你已知的知识。
```
{{knowledge}}
```

你的任务是根据上面给出的知识，回答用户的问题。当你回答时，你的回复必须遵循以下约束：

1. 只回复以上知识中包含的信息。
2. 当你回答问题需要一些额外知识的时候，只能使用非常确定的知识和信息，以确保不会误导用户。
3. 如果你无法确切回答用户问题的答案，请直接回复"不知道"，并给出原因。
4. 使用中文回答。

你需要回答：

{{query}}
'''.strip())

class Chat_View_Base():
    datamodel = SQLAInterface(Chat)
    route_base = '/chat_modelview/api'
    label_title = _('智能体配置')
    base_order = ("id", "desc")
    order_columns = ['id']
    base_filters = [["id", Chat_Filter, lambda: []]]  # 设置权限过滤器

    spec_label_columns = {
        "chat_type": _("交互类型"),
        "hello": _("欢迎语"),
        "tips": _("输入示例"),
        "knowledge": _("知识库"),
        "service_type": _("接口类型"),
        "service_config": _("接口配置"),
        "session_num": _("上下文条数"),
        "prompt": _("提示词模板")
    }

    list_columns = ['name', 'icon', 'label', 'chat_type', 'service_type', 'owner', 'session_num', 'hello']
    cols_width = {
        "name": {"type": "ellip1", "width": 100},
        "label": {"type": "ellip2", "width": 150},
        "chat_type": {"type": "ellip1", "width": 100},
        "hello": {"type": "ellip1", "width": 200},
        "tips": {"type": "ellip1", "width": 200},
        "service_type": {"type": "ellip1", "width": 100},
        "owner": {"type": "ellip1", "width": 200},
        "session_num":{"type": "ellip1", "width": 100},
        "knowledge": {"type": "ellip1", "width": 200},
        "prompt": {"type": "ellip1", "width": 200},
        "service_config": {"type": "ellip1", "width": 200},
    }
    default_icon = '<svg t="1683877543698" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4469" width="50" height="50"><path d="M894.1 355.6h-1.7C853 177.6 687.6 51.4 498.1 54.9S148.2 190.5 115.9 369.7c-35.2 5.6-61.1 36-61.1 71.7v143.4c0.9 40.4 34.3 72.5 74.7 71.7 21.7-0.3 42.2-10 56-26.7 33.6 84.5 99.9 152 183.8 187 1.1-2 2.3-3.9 3.7-5.7 0.9-1.5 2.4-2.6 4.1-3 1.3 0 2.5 0.5 3.6 1.2a318.46 318.46 0 0 1-105.3-187.1c-5.1-44.4 24.1-85.4 67.6-95.2 64.3-11.7 128.1-24.7 192.4-35.9 37.9-5.3 70.4-29.8 85.7-64.9 6.8-15.9 11-32.8 12.5-50 0.5-3.1 2.9-5.6 5.9-6.2 3.1-0.7 6.4 0.5 8.2 3l1.7-1.1c25.4 35.9 74.7 114.4 82.7 197.2 8.2 94.8 3.7 160-71.4 226.5-1.1 1.1-1.7 2.6-1.7 4.1 0.1 2 1.1 3.8 2.8 4.8h4.8l3.2-1.8c75.6-40.4 132.8-108.2 159.9-189.5 11.4 16.1 28.5 27.1 47.8 30.8C846 783.9 716.9 871.6 557.2 884.9c-12-28.6-42.5-44.8-72.9-38.6-33.6 5.4-56.6 37-51.2 70.6 4.4 27.6 26.8 48.8 54.5 51.6 30.6 4.6 60.3-13 70.8-42.2 184.9-14.5 333.2-120.8 364.2-286.9 27.8-10.8 46.3-37.4 46.6-67.2V428.7c-0.1-19.5-8.1-38.2-22.3-51.6-14.5-13.8-33.8-21.4-53.8-21.3l1-0.2zM825.9 397c-71.1-176.9-272.1-262.7-449-191.7-86.8 34.9-155.7 103.4-191 190-2.5-2.8-5.2-5.4-8-7.9 25.3-154.6 163.8-268.6 326.8-269.2s302.3 112.6 328.7 267c-2.9 3.8-5.4 7.7-7.5 11.8z" fill="#2c2c2c" p-id="4470"></path></svg>'

    knowledge_config = '''
{
    "type": "api|file",
    "url":"",   # api请求地址
    "headers": {},  # api请求的附加header
    "data": {},   # api请求的附加data
    "file":"/mnt/$username/",   # 文件地址，或者目录地址，可以多个文件
    "upload_url": "", # 知识库的上传地址
    "recall_url": "", # 召回地址
}
    '''
    service_config = '''
openai接口类型
{
    "llm_url": "",  # 请求的url
    "llm_headers": {
        "xxxxx": "xxxxxx"   # 额外添加的header
    },
    "llm_tokens": [],    # chatgpt的token池
    "llm_data": {
        "xxxxx": "xxxxxx"   # 额外添加的json参数
    },
    "stream": "false" # 是否流式响应
}

aihub接口类型
{
    "url": "aihub应用请求地址",
    "data": {
        "prompt": "$text"   # 输入变量名和输入变量类型
        # $text为用户输入的文本
        # $image为用户输入的图片
        # $audio为用户输入的音频
        # $video为用户输入的视频
    },
    "output": "image",  # 支持text，markdown，image，audio，video
    "req_num":1,     # 请求多少次模型，默认1次，在文生图中可以使用4次
    "stream": "true"
}

        '''

    options_demo={"xAxis":{"type":"category","data":["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]},"yAxis":{"type":"value"},"series":[{"data":[150,230,224,218,135,147,260],"type":"line"}]}
    test_api_resonse = {
        # "text":"这里是文本响应体",
        "echart": json.dumps(options_demo)
    }

    add_fieldsets = [
        (
            _('基础配置'),
            {"fields": ['name','icon','label','doc','owner'], "expanded": True},
        ),
        (
            _('提示词配置'),
            {"fields": ['chat_type','hello','tips','knowledge','prompt','session_num'], "expanded": True},
        ),
        (
            _('模型服务'),
            {"fields": ['service_type','service_config','expand'], "expanded": True},
        )
    ]

    edit_fieldsets=add_fieldsets

    add_form_extra_fields = {
        "name": StringField(
            label= _('名称'),
            description= _('英文名(小写字母、数字、- 组成)，最长50个字符'),
            default='',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$")]
        ),
        "label": StringField(
            label= _('标签'),
            default='',
            description = _('中文名'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "icon": StringField(
            label= _('图标'),
            default=default_icon, # random.choice(icon_choices),
            description= _('svg格式图标，图标宽高设置为50*50，<a target="_blank" href="https://www.iconfont.cn/">iconfont</a>'),
            widget=BS3TextFieldWidget(),
            # choices=[[str(x),Markup(icon_choices[x])] for x in range(len(icon_choices))],
            validators=[DataRequired()]
        ),
        "owner": StringField(
            label= _('责任人'),
            default='*',
            description= _('可见用户，*表示所有用户可见，将责任人列为第一管理员，逗号分割多个责任人'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),

        "chat_type": SelectField(
            label= _('对话类型'),
            description='',
            default='text',
            widget=MySelect2Widget(),
            choices=[['text', _('文本对话')], ['digital_human', _("数字人")]],
            validators=[]
        ),
        "service_type": SelectField(
            label= _('服务类型'),
            description= _('接口类型，并不一定是openai，只需要符合http请求响应格式即可'),
            widget=Select2Widget(),
            default='openai',
            choices=[[x, x] for x in ["openai",'aihub','chatbi','autogpt',_('召回列表')]],
            validators=[]
        ),
        "service_config": StringField(
            label= _('接口配置'),
            default=json.dumps({
                "llm_url": "",
                "llm_tokens": [],
                "stream": "true"
            },indent=4,ensure_ascii=False),
            description= _('接口配置，每种接口类型配置参数不同'),
            widget=MyBS3TextAreaFieldWidget(rows=5, tips=Markup('<pre><code>' + service_config + "</code></pre>")),
            validators=[DataRequired()]
        ),
        "knowledge": StringField(
            label= _('知识库'),
            default=json.dumps({
                "type": "file",
                "file": [__("文件地址")]
            },indent=4,ensure_ascii=False),
            description= _('先验知识配置。如果先验字符串少于1800个字符，可以直接填写字符串，否则需要使用json配置'),
            widget=MyBS3TextAreaFieldWidget(rows=5, tips=Markup('<pre><code>' + knowledge_config + "</code></pre>")),
            validators=[]
        ),
        "prompt":StringField(
            label= _('提示词'),
            default=prompt_default,
            description= _('提示词模板，包含{{knowledge}}知识库召回内容，{{history}}为多轮对话，{{query}}为用户的问题'),
            widget=MyBS3TextAreaFieldWidget(rows=5),
            validators=[]
        ),
        "tips": StringField(
            label= _('输入示例'),
            default='',
            description= _('提示输入，多个提示输入，多行配置'),
            widget=MyBS3TextAreaFieldWidget(rows=3),
            validators=[]
        ),
        "expand": StringField(
            label= _('扩展'),
            default=json.dumps({
                "index":int(time.time())
            },indent=4,ensure_ascii=False),
            description= _('配置扩展参数，"index":控制显示顺序,"isPublic":控制是否为公共应用'),
            widget=MyBS3TextAreaFieldWidget(),
            validators=[]
        ),
    }
    from copy import deepcopy
    edit_form_extra_fields = add_form_extra_fields

    # @pysnooper.snoop()
    def pre_update_web(self, chat=None):
        pass
        self.edit_form_extra_fields['name'] = StringField(
            _('名称'),
            description=_('英文名(小写字母、数字、- 组成)，最长50个字符'),
            default='',
            widget=MyBS3TextFieldWidget(readonly=True if chat else False),
            validators=[DataRequired(), Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$")]
        )

    def pre_add_web(self):
        self.default_filter = {
            "expand": '"isPublic": true'
        }
        self.pre_update_web()

    # 如果传上来的有文件
    # @pysnooper.snoop()
    def pre_add_req(self, req_json=None):
        # 针对chat界面的页面处理
        # chat界面前端，会有files参数
        if req_json and 'files' in req_json:
            expand = json.loads(req_json.get('expand', '{}'))
            name=req_json.get('name','')
            # 在添加的时候做一些特殊处理
            if request.method=='POST':
                name = f'{g.user.username}-faq-{uuid.uuid4().hex[:4]}'
                req_json['name'] = name
                req_json['hello'] = __('自动为您创建的私人对话，不使用上下文，左下角可以清理会话和修改知识库配置')
                req_json['session_num'] = '0'
                req_json['icon'] = default_icon

            files_path = []
            files = req_json['files']
            if type(files) != list:
                files = [files]

            exist_knowledge = {}
            if name:
                chat = db.session.query(Chat).filter_by(name=name).first()
                if chat:
                    try:
                        exist_knowledge = json.loads(chat.knowledge)
                    except:
                        exist_knowledge = {}

            file_arr = []
            for file in files:
                file_name = file.get('name', '')
                file_type = file.get("type", '')
                file_content = file.get("content", '')   # 最优最新一次上传的才有这个。
                file_arr.append({
                    "name": file_name,
                    "type": file_type
                })
                # 拼接文件保存路径
                file_path = f'/data/k8s/kubeflow/global/knowledge/{name}/{file_name}'
                files_path.append(file_path)
                # 如果有新创建的文件内容
                if file_content:
                    content = base64.b64decode(file_content)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    file = open(file_path, mode='wb')
                    file.write(content)
                    file.close()


            knowledge = {
                "type": "file",
                "file": files_path,
                # "file_arr": file_arr,
                "upload_url": "http://chat-embedding.aihub:80/aihub/chat-embedding/api/upload_files",
                "recall_url": "http://chat-embedding.aihub:80/aihub/chat-embedding/api/recall"
            }
            expand['fileMetaList']=file_arr

            if exist_knowledge.get('status',''):
                knowledge['status']=exist_knowledge.get('status','')
                knowledge['update_time'] = exist_knowledge.get('update_time','')
                knowledge['upload_url'] = exist_knowledge.get('upload_url', '')
                knowledge['recall_url'] = exist_knowledge.get('recall_url', '')

            req_json['knowledge'] = json.dumps(knowledge, indent=4, ensure_ascii=False)
            req_json['expand']=json.dumps(expand, indent=4, ensure_ascii=False)
            del req_json['files']

        return req_json

    pre_update_req = pre_add_req

    # @pysnooper.snoop(watch_explode=('req_json',))
    # def pre_update_req(self, req_json=None):
    #     print(g.user.username)
    #     owner = req_json.get('owner','')
    #     if g.user.username in owner:
    #         self.pre_add_req(req_json)
    #     else:
    #         flash('只有创建者或管理员可以配置', 'warning')
    #         raise MyappException('just creator can add/edit')

    # @pysnooper.snoop(watch_explode=('item',))
    def pre_add(self, item):
        if not item.knowledge or not item.knowledge.strip():
            item.knowledge = '{}'
        if not item.owner:
            item.owner = g.user.username
        if not item.icon:
            item.icon = default_icon # random.choice(icon_choices)
        if not item.chat_type:
            item.chat_type = 'text'
        if not item.service_type:
            item.service_type = 'openai'
        if not item.service_config or not item.service_config.strip():
            service_config = {
                "llm_url": "",
                "llm_tokens": [],
                "stream": "true"
            }
            item.service_config = json.dumps(service_config)

        service_config = json.loads(item.service_config) if item.service_config.strip()else {}
        expand = json.loads(item.expand) if item.expand.strip() else {}
        knowledge = json.loads(item.knowledge) if item.knowledge.strip() else {}

        # 配置扩展字段
        if item.expand and item.expand.strip():
            item.expand=json.dumps(json.loads(item.expand),indent=4,ensure_ascii=False)
        try:
            expand = json.loads(item.expand) if item.expand else {}
            expand['isPublic'] = expand.get('isPublic',True)
            # 把之前的属性更新上，避免更新的时候少填了什么属性
            src_expand = self.src_item_json.get("expand",'{}')
            if src_expand:
                src_expand = json.loads(src_expand)
                src_expand.update(expand)
                expand = src_expand
            item.expand = json.dumps(expand, indent=4, ensure_ascii=False)
        except Exception as e:
            print(e)

        # 如果是私有应用，添加一些file_arr
        if not expand.get('isPublic', True):
            fileMetaList = expand.get('fileMetaList',[])
            files = knowledge.get('file',[])
            # 如果有文件，但是没有文件属性信息，则更新
            if not fileMetaList and files:
                expand['fileMetaList'] = []
                if type(files)!=list:
                    files = [files]
                for file in files:
                    name = os.path.basename(file)

                    if '.' in name:
                        ext = name[name.rindex('.')+1:]
                        file_map={
                            "map":"application/octet-stream",
                            "csv":"text/csv",
                            "pdf":"application/pdf",
                            "txt":"text/plain"
                        }
                        file_attr = {
                            "name": name,
                            "type": file_map[ext]
                        }
                        expand['fileMetaList'].append(file_attr)



            # 如果有知识库
            if knowledge.get('file',[]) or knowledge.get('url',''):
                item.prompt = prompt_default
            # 如果没有，就自动多轮对话
            else:
                knowledge['status']='在线'
                knowledge['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                item.session_num=10
                item.prompt = '''
{{history}}
Human:{{query}}
AI:
'''.strip()


        # item.icon = item.icon.replace('width="200"','width="50"').replace('height="200"','height="50"')
        item.icon = re.sub(r'width="\d+(\.\d+)?(px)?"', f'width="50px"', item.icon)
        item.icon = re.sub(r'height="\d+(\.\d+)?(px)?"', f'height="50px"', item.icon)
        if '{{query}}' not in item.prompt:
            item.prompt = item.prompt+"\n{{query}}\n"

        item.service_config = json.dumps(service_config, indent=4, ensure_ascii=False)
        item.expand = json.dumps(expand, indent=4, ensure_ascii=False)
        try:
            item.knowledge = json.dumps(knowledge, indent=4, ensure_ascii=False)
        except Exception as e:
            print(e)

    # @pysnooper.snoop()
    def pre_update(self, item):
        if g.user.username in self.src_item_json.get('owner','') or g.user.is_admin():
            self.pre_add(item)
        else:
            flash(__('只有创建者或管理员可以配置'), 'warning')
            raise MyappException('just creator can add/edit')

    # @pysnooper.snoop(watch_explode=('item',))
    def post_add(self, item):
        try:
            if not self.src_item_json:
                self.src_item_json = {}

            src_file = json.loads(self.src_item_json.get('knowledge', '{}')).get("file", '')
            last_time = json.loads(self.src_item_json.get('knowledge', '{}')).get("update_time",'')
            if last_time:
                last_time = datetime.datetime.strptime(last_time,'%Y-%m-%d %H:%M:%S')

            knowledge_config = json.loads(item.knowledge) if item.knowledge else {}
            exist_file = knowledge_config.get("file", '')
            # 文件变了，或者更新时间过期了，都要重新更新
            if exist_file and (src_file != exist_file or not last_time or (datetime.datetime.now()-last_time).total_seconds()>3600):
                self.upload_knowledge(chat=item, knowledge_config=knowledge_config)

        except Exception as e:
            print(e)

    def post_update(self, item):
        self.post_add(item)

    # 按配置的索引进行排序
    def post_list(self, items):
        from myapp.utils import core
        return core.sort_expand_index(items)
        # print(_response['data'])
        # _response['data'] = sorted(_response['data'],key=lambda chat:float(json.loads(chat.get('expand','{}').get("index",1))))

    @action("copy", "复制", confirmation= '复制所选记录?', icon="fa-copy", multiple=True, single=True)
    def copy(self, chats):
        if not isinstance(chats, list):
            chats = [chats]
        try:
            for chat in chats:
                new_chat = chat.clone()
                new_chat.name = new_chat.name+"-copy"
                new_chat.created_on = datetime.datetime.now()
                new_chat.changed_on = datetime.datetime.now()
                db.session.add(new_chat)
                db.session.commit()
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            print(e)
            raise e

        return redirect(request.referrer)

    @expose('/chat/<chat_name>', methods=['POST', 'GET'])
    # @pysnooper.snoop()
    def chat(self, chat_name, args=None):
        if chat_name == 'chatbi':
            files = os.listdir('myapp/utils/echart/')
            files = ['area-stack.json', 'rose.json', 'mix-line-bar.json', 'pie-nest.json', 'bar-stack.json',
                   'candlestick-simple.json', 'graph-simple.json', 'tree-polyline.json', 'sankey-simple.json',
                   'radar.json', 'sunburst-visualMap.json', 'parallel-aqi.json', 'funnel.json',
                   'sunburst-visualMap.json', 'scatter-effect.json']
            files = [os.path.join('myapp/utils/echart/',file) for file in files if '.json' in file]

            return {
                "status": 0,
                "finish": False,
                "message": 'success',
                "result": [
                    {
                        "text":"未配置后端模型，这里生成示例看板\n\n",
                        # "echart": json.dumps(options_demo)
                        "echart":open(random.choice(files)).read()
                    }
                ]
            }

        if not args:
            args = request.get_json(silent=True)
        if not args:
            args = {}
        session_id = args.get('session_id', 'xxxxxx')
        request_id = args.get('request_id', str(datetime.datetime.now().timestamp()))
        search_text = args.get('search_text', '')
        search_audio = args.get('search_audio', None)
        search_image = args.get('search_image', None)
        search_video = args.get('search_video', None)
        username = args.get('username', '')
        enable_tts = args.get('enable_tts', False)
        if not username:
            username = g.user.username
        if g:
            g.after_message=''
        stream = args.get('stream', False)
        if str(stream).lower()=='false':
            stream = False

        begin_time = datetime.datetime.now()

        chat = db.session.query(Chat).filter_by(name=chat_name).first()
        if not chat:
            return jsonify({
                "status": 1,
                "message": __('聊天不存在'),
                "result": []
            })

        # 如果超过一定聊天数目，则禁止
        # if username not in conf.get('ADMIN_USER').split(','):
        #     log_num = db.session.query(ChatLog).filter(ChatLog.username==username).filter(ChatLog.answer_status=='成功').filter(ChatLog.created_on>datetime.datetime.now().strftime('%Y-%m-%d')).all()
        #     if len(log_num)>10:
        #         return jsonify({
        #             "status": 1,
        #             "finish": 0,
        #             "message": '聊天次数达到上限，每人，每天仅限10次',
        #             "result": [{"text":"聊天次数达到上限，每人，每天仅限10次"}]
        #         })

        stream_config = json.loads(chat.service_config).get('stream', True)
        if stream_config==False or str(stream_config).lower() == 'false':
            stream = False

        enable_history = args.get('history', True)
        chatlog=None
        # 添加数据库记录
        try:
            text = emoji.demojize(search_text)
            search_text = re.sub(':\S+?:', ' ', text)  # 去除表情
            chatlog = ChatLog(
                username=str(username),
                chat_id=chat.id,
                query=search_text,
                answer="",
                manual_feedback="",
                answer_status="created",
                answer_cost='0',
                err_msg="",
                created_on=str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                changed_on=str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            )
            db.session.add(chatlog)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(e)

        return_result = []
        return_message = ''
        return_status = 1
        finish = ''
        answer_status = 'making'
        err_msg = ''
        history = []
        try:
            if enable_history and int(chat.session_num):
                history = cache.get('chat_' + session_id)  # 所有需要的上下文
                if not history:
                    history = []
        except Exception as e:
            print(e)

        if chat.service_type.lower() == 'openai' or chat.service_type.lower() == 'chatgpt4' or chat.service_type.lower() == 'chatgpt3.5':
            if stream:
                if chatlog:
                    chatlog.answer_status = 'push chatgpt'
                    db.session.commit()
                res = self.chatgpt(
                    chat=chat,
                    session_id=session_id,
                    search_text=search_text,
                    enable_history=enable_history,
                    history=history,
                    chatlog_id=chatlog.id,
                    stream=True
                )
                if chatlog:
                    chatlog.answer_status = '成功'
                    db.session.commit()
                return res

            else:
                if chatlog:
                    chatlog.answer_status = 'push chatgpt'
                    db.session.commit()

                return_status, text = self.chatgpt(
                    chat=chat,
                    session_id=session_id,
                    search_text=search_text,
                    enable_history=enable_history,
                    history=history,
                    chatlog_id=chatlog.id,
                    stream=False
                )
                return_message = __('失败') if return_status else __("成功")
                answer_status = return_message
                return_result = [
                    {
                        "text": text
                    }
                ]


        if chat.service_type.lower() == 'openai' or chat.service_type.lower() == 'chatgpt4' or chat.service_type.lower() == 'chatgpt3.5':
            if stream:
                if chatlog:
                    chatlog.answer_status = 'push chatgpt'
                    db.session.commit()
                res = self.chatgpt(
                    chat=chat,
                    session_id=session_id,
                    search_text=search_text,
                    enable_history=enable_history,
                    history=history,
                    chatlog_id=chatlog.id,
                    stream=True
                )
                if chatlog:
                    chatlog.answer_status = __('成功')
                    db.session.commit()
                return res

            else:
                if chatlog:
                    chatlog.answer_status = 'push chatgpt'
                    db.session.commit()

                return_status, text = self.chatgpt(
                    chat=chat,
                    session_id=session_id,
                    search_text=search_text,
                    enable_history=enable_history,
                    history=history,
                    chatlog_id=chatlog.id,
                    stream=False
                )
                return_message = __('失败') if return_status else __("成功")
                answer_status = return_message
                return_result = [
                    {
                        "text": text
                    }
                ]

        # 仅返回召回列表
        if __('召回列表') in chat.service_type.lower():
            knowledge = self.get_remote_knowledge(chat, search_text,score=True)
            knowledge = [__("内容：\n\n    ") + x['context'].replace('\n','\n    ') + "\n\n" + __("得分：\n\n    ") + str(x.get('score', '')) + "\n\n" + __("文件：\n\n    ") + str(x.get('file', '')) for x in knowledge]
            if knowledge:
                text = '\n\n-------\n'.join(knowledge)
                return_message = __("成功")
                answer_status = return_message
                return_result = [
                    {
                        "text": text
                    }
                ]
            else:
                return_result = [
                    {
                        "text": __('召回内容为空')
                    }
                ]

        # 多轮召回方式
        if __('多轮') in chat.service_type.lower():
            knowledge = self.get_remote_knowledge(chat, search_text)
            if knowledge:
                return_message = __("成功")
                answer_status = return_message
                return_result = [
                    {
                        "text": text
                    } for text in knowledge
                ]
            else:
                return_result = [
                    {
                        "text": __('未找到相关内容')
                    }
                ]

        if chat.service_type.lower() == 'aihub':
            return_status, return_res = self.aigc4(chat=chat, search_text=search_text)
            if not return_status:
                return return_res

        # 添加数据库记录
        if chatlog:
            try:
                canswar = "\n".join(item.get('text','') for item in return_result)
                chatlog.query = search_text
                # chatlog.answer = canswar   # 内容太多了
                chatlog.answer_cost = str((datetime.datetime.now()-begin_time).total_seconds())
                chatlog.answer_status=answer_status,
                chatlog.err_msg = return_message
                db.session.commit()

                # 正确响应的话，才设置为历史状态
                if history != None and not return_status:
                    history.append((search_text, canswar))
                    history = history[0 - int(chat.session_num):]
                    try:
                        cache.set('chat_' + session_id, history, timeout=300)   # 人连续对话的时间跨度
                    except Exception as e:
                        print(e)

            except Exception as e:
                db.session.rollback()
                print(e)

        return {
            "status": return_status,
            "finish": finish,
            "message": return_message,
            "result": [x for x in return_result if x]
        }

    # @pysnooper.snoop()
    def upload_knowledge(self,chat,knowledge_config):
        """
        上传文件到远程服务
        @param chat: 场景对象
        @param knowledge_config: 知识库配置
        @return:
        """
        # 没有任何值就是空的
        files=[]
        if not knowledge_config:
            return ''

        knowledge_type = knowledge_config.get("type", 'str')
        if knowledge_type == 'str':
            knowledge = knowledge_config.get("content", '')
            if knowledge:
                file_path = f'knowledge/{chat.name}/{str(time.time()*1000)}'
                os.makedirs(os.path.dirname(file_path),exist_ok=True)
                file = open(file_path,mode='w')
                file.write(knowledge)
                file.close()
                files.append(file_path)

        if knowledge_type == 'api':
            url = knowledge_config.get("url", '')
            if not url:
                return ''
            headers = knowledge_config.get("headers", {})
            data = knowledge_config.get("data", {})
            if data:
                res = requests.post(url, headers=headers, json=data,verify=False)
            else:
                res = requests.get(url, headers=headers,verify=False)

            if res.status_code == 200:
                # 获取文件名和文件格式
                filename = os.path.basename(url)
                file_format = os.path.splitext(filename)[1]

                # 拼接文件保存路径
                file_path = f'knowledge/{chat.name}/{str(time.time() * 1000)}'
                if file_format:
                    file_path = file_path+"."+file_format
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file = open(file_path, mode='wb')
                file.write(res.content)
                file.close()
                files.append(file_path)

        if knowledge_type == 'file':
            file_paths = knowledge_config.get("file", '')
            if type(file_paths)!=list:
                file_paths = [file_paths]
            for file_path in file_paths:
                if re.match('^/mnt', file_path):
                    file_path = "/data/k8s/kubeflow/pipeline/workspace" + file_path.replace("/mnt", '')
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        files.append(file_path)
                    if os.path.isdir(file_path):
                        for root, dirs_temp, files_temp in os.walk(file_path):
                            for name in files_temp:
                                one_file_path = os.path.join(root, name)
                                # print(one_file_path)
                                if os.path.isfile(one_file_path):
                                    files.append(one_file_path)

        if knowledge_type == 'sql':
            return ''


        service_config = json.loads(chat.service_config)
        upload_url = knowledge_config.get("upload_url", '')
        if files:
            if '127.0.0.1' in request.host_url:
                print('发现的知识库文件：',files)
                knowledge = json.loads(chat.knowledge) if chat.knowledge else {}
                knowledge['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                knowledge['status'] = "uploading"
                chat.knowledge = json.dumps(knowledge,indent=4,ensure_ascii=False)
                db.session.commit()
                files_content = [('files', (os.path.basename(file), open(file, 'rb'))) for file in files]
                data = {"chat_id": chat.name}
                response = requests.post(upload_url, files=files_content, data=data,verify=False)
                print('上传私有知识响应：',json.dumps(json.loads(response.text), ensure_ascii=False, indent=4))
                knowledge['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                knowledge['status'] = "online"
                chat.knowledge = json.dumps(knowledge, indent=4, ensure_ascii=False)
                db.session.commit()

            else:
                from myapp.tasks.async_task import upload_knowledge
                kwargs = {
                    "files": files,
                    "chat_id":chat.name,
                    "upload_url":upload_url,
                    # 可以根据不同的配置来决定对数据做什么处理，比如
                    # "config":{
                    #     "file":{
                    #         "cube-studio.csv":{
                    #             "embedding_columns": ["问题"],
                    #             "llm_columns": ['问题', '答案'],
                    #             "keywork_columns": [],
                    #             "summary_columns": []
                    #         }
                    #     }
                    # }
                }
                upload_knowledge.apply_async(kwargs=kwargs)


    all_chat_knowledge = {}
    # 根据配置获取远程的先验知识
    # @pysnooper.snoop()
    def get_remote_knowledge(self,chat,search_text,score=False):
        """
        召回服务
        @param chat: 场景对象
        @param search_text: 搜索文本
        @return: 获取召回的前3个文本
        """
        knowledge=[]
        try:
            service_config = json.loads(chat.service_config)
            knowledge_config = json.loads(chat.knowledge)

            # 时间过时就发过去重新更新知识库
            update_time = knowledge_config.get("update_time",'')
            if update_time:
                update_time = datetime.datetime.strptime(update_time,'%Y-%m-%d %H:%M:%S')
            if not update_time or (datetime.datetime.now()-update_time).total_seconds()>3600 or knowledge_config.get("status","")!='在线':
                self.upload_knowledge(chat=chat,knowledge_config=knowledge_config)


            # 进行召回
            recall_url = knowledge_config.get("recall_url", '')
            if recall_url:
                data={
                    "knowledge_base_id":chat.name,
                    "question":search_text,
                    "history":[]
                }
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                res = requests.post(recall_url,json=data,headers=headers,timeout=5,verify=False)
                if res.status_code==200:
                    recall_result = res.json()
                    print('召回响应',json.dumps(recall_result,indent=4,ensure_ascii=False))
                    if 'result' in recall_result:
                        knowledge= recall_result['result']
                        if not score:
                            knowledge = [x['context'] for x in knowledge]
                        knowledge = knowledge[0:3]
                    else:
                        source_documents = recall_result.get('source_documents',[])
                        # source_documents = sorted(source_documents,key=lambda item:float(item.get('score',1)))  # 按分数值排序，应该走排序算法
                        source_documents = source_documents[:3]  # 只去前面3个
                        all_sources = []
                        for index,item in enumerate(source_documents):
                            if int(item.get('score', 1)) > float(knowledge_config.get("min_score",0)):   # 根据最小分数值确定
                                source = item.get('source', '')
                                if source:
                                    all_sources.append(source)

                                knowledge.append(item['context'])
                        all_sources = [x.strip() for x in list(set(all_sources)) if x.strip()]
                        after_message = ''
                        if all_sources:
                            for index,source in enumerate(all_sources):
                                source_url = request.host_url.rstrip('/') + f"/aitalk_modelview/api/file/{chat.name}/" + source.lstrip('/')
                                after_message += f'[文档{index}]({source_url}) '
                            # g.after_message = g.after_message + f"\n\n{after_message}"

        except Exception as e:
            print(e)

        return knowledge


    # @pysnooper.snoop()
    # 获取header和url
    def get_llm_url_header(self,chat,stream=False):
        """
        获取访问地址和有效token
        @param chat:
        @param stream:
        @return:
        """
        url = json.loads(chat.service_config).get("llm_url", '')
        headers = json.loads(chat.service_config).get("llm_headers", {})
        if stream:
            headers['Accept'] = 'text/event-stream'
        else:
            headers['Accept'] = 'application/json'

        if not url:
            llm_url = conf.get('CHATGPT_CHAT_URL', 'https://api.openai.com/v1/chat/completions')
            if llm_url:
                if type(llm_url) == list:
                    llm_url = random.choice(llm_url)
                else:
                    llm_url = llm_url
                url=llm_url

        llm_tokens = json.loads(chat.service_config).get("llm_tokens", [])
        llm_token = ''
        if llm_tokens:
            if type(llm_tokens) != list:
                llm_tokens = [llm_tokens]
            # 如果有过多错误的token，则直接废弃
            error_token = json.loads(chat.service_config).get("miss_tokens",{})
            if error_token:
                right_llm_tokens= [token for token in llm_token if int(error_token.get(token,0))<100]
                if right_llm_tokens:
                    llm_tokens=right_llm_tokens

            llm_token = random.choice(llm_tokens)
            headers['Authorization'] = 'Bearer ' + llm_token  # openai的接口
            headers['api-key'] = llm_token   # 微软的接口
        else:
            llm_tokens = conf.get('CHATGPT_TOKEN','')
            if llm_tokens:
                if type(llm_tokens)==list:
                    llm_token = random.choice(llm_tokens)
                else:
                    llm_token = llm_tokens
                headers['Authorization'] = 'Bearer ' + llm_token    # openai的接口
                headers['api-key']=llm_token    # 微软的接口

        return url,headers,llm_token

    # 组织提问词
    # @pysnooper.snoop(watch_explode=('system_content'))
    def generate_prompt(self,chat, search_text, enable_history, history=[]):
        messages = chat.prompt

        messages = messages.replace('{{query}}', search_text)

        # 获取知识库
        if '{{knowledge}}' in chat.prompt:
            knowledge = chat.knowledge  # 直接使用原文作为知识库
            try:
                knowledge_config = json.loads(chat.knowledge)
                try:
                    knowledge = self.get_remote_knowledge(chat, search_text)
                except Exception as e1:
                    print(e1)
            except Exception as e:
                print(e)

            if type(knowledge) != list:
                knowledge = [str(knowledge)]
            knowledge = [x for x in knowledge if x.strip()]
            # 拼接请求体
            print('召回知识库', json.dumps(knowledge, indent=4, ensure_ascii=False))
            added_knowledge = []
            # 添加私有知识库，要满足token限制
            for item in knowledge:
                # 至少要保留前置语句，后置语句，搜索语句。
                if sum([len(x) for x in added_knowledge]) < (max_len - len(messages) - len(item)):
                    added_knowledge.append(item)
            added_knowledge = '\n\n'.join(added_knowledge)
            messages = messages.replace('{{knowledge}}', added_knowledge)

        if '{{history}}' in chat.prompt:

            # 拼接上下文
            # 先从后往前加看看个数是不是超过了门槛
            added_history=[]
            if enable_history and history:
                for index in range(len(history) - 1, -1, -1):
                    faq = history[index]
                    added_faq="Human: %s\nAI: %s"%(faq[0],faq[1])
                    added_history_len = sum([len(x) for x in added_faq])
                    if len(added_faq) < (max_len-len(messages)-added_history_len):
                        added_history.insert(0,added_faq)
                    else:
                        break
            added_history = '\n'.join(added_history)
            messages = messages.replace('{{history}}', added_history)
        print(messages)
        return [{'role': 'user', 'content': messages}]

    # 生成openai相应格式
    def make_openai_res(self,message,stream=True):
        back = {
            "id": "chatcmpl-7OPUNz80uRGVKLcBMW8aKZT9dg938",
            "object": "chat.completion.chunk" if stream else 'chat.completion',
            "created": int(time.time()),
            "model": "gpt-4-turbo-2024-04-09",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "role": "assistant",
                        "content":message
                    },
                    "message":{
                        "role": "assistant",
                        "content": message
                    }
                }
            ],
            "usage": None
        }
        return json.dumps(back)

    @expose('/chat/chatgpt/<chat_name>', methods=['POST', 'GET'])
    # @pysnooper.snoop()
    def chatgpt_api(self, chat_name):
        """
        为调用chatgpt单独提供的接口
        @param chat_name:
        @return:
        """
        args = request.get_json(silent=True)
        chat = db.session.query(Chat).filter_by(name=chat_name).first()

        session_id = args.get('session_id', 'xxxxxx')
        request_id = args.get('request_id', str(datetime.datetime.now().timestamp()))
        search_text = args.get('search_text', '')

        return_status, text = self.chatgpt(
            chat=chat,
            session_id=session_id,
            search_text=search_text,
            enable_history=False,
            history=[],
            chatlog_id=None,
            stream=False
        )
        return jsonify({
            "status": return_status,
            "message": __('失败') if return_status else __("成功"),
            "result": [
                {
                    "text": text
                }
            ]
        })

    # 调用chatgpt接口
    # @pysnooper.snoop()
    def chatgpt(self, chat, session_id, search_text, enable_history,history=[], chatlog_id=None, stream=True):
        max_retry=3
        for i in range(0,max_retry):

            url, headers, llm_token = self.get_llm_url_header(chat, stream)
            message = self.generate_prompt(chat=chat, search_text=search_text, enable_history=enable_history, history=history)
            service_config = json.loads(chat.service_config)
            data = {
                'model': 'gpt-4-turbo-2024-04-09',
                'messages': message,
                'temperature': service_config.get("temperature",1),  # 问答发散度 0-2 越高越发散 较高的值（如0.8）将使输出更随机，较低的值（如0.2）将使其更集中和确定性
                'top_p': service_config.get("top_p",0.5),  # 同temperature，如果设置 0.1 意味着只考虑构成前 10% 概率质量的 tokens
                'n': 1,  # top n可选值
                'stream': stream,
                'stop': 'elit proident sint',  #
                'max_tokens': service_config.get("max_tokens",2500),  # 最大返回数
                'presence_penalty': service_config.get("presence_penalty",1),  # [控制主题的重复度]，-2.0（抓住一个主题使劲谈论） ~ 2.0（最大程度避免谈论重复的主题） 之间的数字，正值会根据到目前为止是否出现在文本中来惩罚新 tokens，从而增加模型谈论新主题的可能性
                'frequency_penalty': 0, # [重复度惩罚因子], -2.0(可以尽情出现相同的词汇) ~ 2.0 (尽量不要出现相同的词汇)
                'user': 'user',
            }
            data.update(json.loads(chat.service_config).get("llm_data", {}))


            if stream:
                # 返回流响应
                import sseclient

                res = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    stream=stream,
                    verify=False
                )
                if res.status_code != 200 and i<(max_retry-1):
                    continue
                client = sseclient.SSEClient(res)

                # @pysnooper.snoop(watch_explode='message')
                def generate(history):

                    back_message = ''
                    for event in client.events():
                        message = event.data
                        finish = False
                        if message != '[DONE]':
                            choices = json.loads(event.data)['choices']
                            if choices:
                                message = choices[0].get('delta', {}).get('content', '')
                            else:
                                message=''
                            print(message, flush=True, end='')
                        # print(message)
                        if message == '[DONE]':
                            finish = True
                            back_message = back_message+g.after_message
                            if chatlog_id:
                                chatlog = db.session.query(ChatLog).filter_by(id=int(chatlog_id)).first()
                                chatlog.answer_status = '成功'
                                # chatlog.answer = back_message  # 内容太多了
                                db.session.commit()
                                if history != None:
                                    history.append((search_text, back_message))
                                    history = history[0 - int(chat.session_num):]
                                    try:
                                        cache.set('chat_' + session_id, history, timeout=300)  # 人连续对话的时间跨度
                                    except Exception as e:
                                        print(e)
                        else:
                            back_message = back_message + message
                        # 随机乱码，用来避免内容中包含此内容，实现每次返回内容的分隔
                        back = "TQJXQKT0POF6P4D:" + json.dumps(
                            {
                                "message": "success",
                                "status": 0,
                                "finish":finish,
                                "result": [
                                    {"text": back_message},
                                ]
                            }, ensure_ascii=False
                        ) + "\n\n"
                        yield back

                response = Response(stream_with_context(generate(history=history if enable_history else None)),mimetype='text/event-stream')
                response.headers["Cache-Control"] = "no-cache"
                response.headers["Connection"] = 'keep-alive'
                response.status_code = res.status_code
                if response.status_code ==401:
                    service_config = json.loads(chat.service_config)
                    # if 'miss_tokens' not in service_config:
                    #     service_config['miss_tokens']={}
                    # service_config['miss_tokens'][llm_token]=service_config['miss_tokens'].get(llm_token,0)+1
                    chat.service_config = json.dumps(service_config,ensure_ascii=False,indent=4)
                    db.session.commit()

                return response

                # 返回普通响应
            else:
                # print(url)
                # print(headers)
                # print(data)
                res = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    verify=False
                )
                if res.status_code != 200 and i < (max_retry - 1):
                    continue
                if res.status_code == 200 or res.status_code == 201:
                    # print(res.text)
                    mes = res.json()['choices'][0]['message']['content']
                    print(mes)
                    return 0, mes
                else:
                    service_config = json.loads(chat.service_config)
                    # if 'miss_tokens' not in service_config:
                    #     service_config['miss_tokens'] = {}
                    # service_config['miss_tokens'][llm_token] = service_config['miss_tokens'].get(llm_token,0) + 1
                    chat.service_config = json.dumps(service_config, ensure_ascii=False, indent=4)
                    db.session.commit()
                    return 1, f'请求{url}失败'

    # @pysnooper.snoop()
    def aigc4(self, chat, search_text):
        """
        aigc 文本转图片
        @param chat:
        @param search_text:
        @return:
        """
        try:
            url = json.loads(chat.service_config).get("aigc_url", '')
            if 'http:' not in url and 'https://' not in url:
                url = urllib.parse.urljoin(request.host_url, url)
            pic_num = json.loads(chat.service_config).get("pic_num", 4)
            headers = json.loads(chat.service_config).get("aigc_headers", {})
            data = {
                "text": search_text,
                "prompt": search_text,
                "steps":50
            }
            data.update(json.loads(chat.service_config).get("aigc_data", {}))
            # @pysnooper.snoop()
            from myapp.utils.core import pic2html
            def generate():
                all_result_image = []
                for i in range(pic_num):
                    # 示例输入
                    time.sleep(1)
                    status, image = 0,f'https://cube-studio.oss-cn-hangzhou.aliyuncs.com/aihub/aigc/aigc{i+1}.jpeg'

                    if not status:
                        all_result_image.append(image)

                    back_message = "未配置后端模型，为您生成4张示例图片：\n"+pic2html(all_result_image,pic_num)
                    # print(back_message)

                    back = "TQJXQKT0POF6P4D:" + json.dumps(
                        {
                            "message": "success",
                            "status": 0,
                            "finish": False,
                            "result": [
                                {"text": back_message},
                            ]
                        }, ensure_ascii=False
                    ) + "\n\n"
                    yield back

            response = Response(stream_with_context(generate()),mimetype='text/event-stream')
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Connection"] = 'keep-alive'
            return 0,response

        except Exception as e:
            return 1, 'aigc报错：' + str(e)

# 添加api
class Chat_View(Chat_View_Base, MyappModelRestApi):


    datamodel = SQLAInterface(Chat)

# 添加api
class Chat_View_Api(Chat_View_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Chat)
    route_base = '/aitalk_modelview/api'
    list_columns = ['id','name', 'icon', 'label', 'chat_type', 'service_type', 'owner', 'session_num', 'hello', 'tips','knowledge','service_config','expand']

    # info接口响应修正
    # @pysnooper.snoop()
    def pre_list_res(self, _response):

        # 把提示语进行分割
        for chat in _response['data']:
            chat['tips'] = [x for x in chat['tips'].split('\n') if x] if chat['tips'] else []
            try:
                service_config = chat.get('service_config', '{}')
                if service_config:
                    chat['service_config'] = json.loads(service_config)
            except Exception as e:
                print(e)
            try:
                knowledge = chat.get('knowledge', '{}')
                if knowledge:
                    chat['knowledge'] = json.loads(knowledge)
            except Exception as e:
                print(e)
            try:
                expand = chat.get('expand', '{}')
                if expand:
                    chat['expand'] = json.loads(expand)
            except Exception as e:
                print(e)
        return _response

appbuilder.add_api(Chat_View)
appbuilder.add_api(Chat_View_Api)
