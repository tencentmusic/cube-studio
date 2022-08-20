"""Contains the logic to create cohesive forms on the explore view"""
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
from wtforms import Field
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from wtforms import widgets
from myapp import app

conf = app.config


from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp,ValidationError
# from myapp.models.base import MyappModelBase
# model_base=MyappModelBase()
#
# class StringField(Field):
#     """
#     This field is the base for most of the more complicated fields, and
#     represents an ``<input type="text">``.
#     """
#     widget = widgets.TextInput()
#
#
#     def __init__(self, label=None, validators=None, filters=tuple(),
#                  description='', id=None, default=None, widget=None,
#                  render_kw=None, _form=None, _name=None, _prefix='',
#                  _translations=None, _meta=None):
#         label=_(model_base.lab('server')) if label==None  else label
#         default='' if default==None else default
#         widget=BS3TextFieldWidget() if widget==None else widget
#         description=description if description else ''
#         validators = [] if validators==None else validators
#
#         return super(StringField, self).__init__(label=label,default=default,widget=widget,validators=validators,description=description)
#
#     # @pysnooper.snoop()
#     def process_formdata(self, valuelist):
#         # aa = self.data
#         if not self.data:
#             self.data=''
#         if valuelist:
#             self.data = valuelist[0]
#         elif self.data is None:
#             self.data = ''
#         aa = self.data
#
#     # @pysnooper.snoop()
#     def _value(self):
#         return text_type(self.data) if self.data is not None else ''


# 处理完再校验
class JsonValidator(object):
    def __init__(self):
        pass

    def __call__(self, form, field):
        data = field.data
        if data is None:
            raise ValidationError('input must json')
        try:
            json.loads(data)
        except Exception as e:
            raise ValidationError("JSON is not valid :%s"%str(e))

class MyCommaSeparatedListField(Field):
    widget = BS3TextFieldWidget()

    def _value(self):
        if self.data:
            return u", ".join(self.data)
        else:
            return u""

    def process_formdata(self, valuelist):
        if valuelist:
            self.data = [x.strip() for x in valuelist[0].split(",")]
        else:
            self.data = []


def filter_not_empty_values(value):
    """Returns a list of non empty values or None"""
    if not value:
        return None
    data = [x for x in value if x]
    if not data:
        return None
    return data




import pysnooper,datetime,time,json
from wtforms.widgets.core import HTMLString,html_params

try:
    from html import escape
except ImportError:
    from cgi import escape
from wtforms.compat import text_type, iteritems



class MyCodeArea(object):
    def __init__(self, code=''):
        self.code=code
    def __call__(self, field, **kwargs):
        if self.code:
            return HTMLString('<pre><code>%s</code></pre>' % (self.code,))
        else:
            return HTMLString('<pre><code>%s</code></pre>' % (field._value(),))
        # return HTMLString('<pre><code>%s</code></pre>' % (field._value(),))

from wtforms import widgets
class MyBS3TextAreaFieldWidget(widgets.TextArea):
    def __init__(self, rows=3,readonly=0):
        self.rows=rows
        self.readonly = readonly
        return super(MyBS3TextAreaFieldWidget, self).__init__()
    def __call__(self, field, **kwargs):
        kwargs["class"] = u"form-control"
        kwargs["rows"] = self.rows
        if field.label:
            kwargs["placeholder"] = field.label.text
        if self.readonly:
            kwargs['readonly']='readonly'
        return super(MyBS3TextAreaFieldWidget, self).__call__(field, **kwargs)


class MyBS3TextFieldWidget(widgets.TextInput):
    def __init__(self, value='',readonly=0):
        self.value=value
        self.readonly = readonly
        return super(MyBS3TextFieldWidget, self).__init__()

    def __call__(self, field, **kwargs):
        kwargs["class"] = u"form-control"
        if field.label:
            kwargs["placeholder"] = field.label.text
        if "name_" in kwargs:
            field.name = kwargs["name_"]
        if self.value:
            kwargs['value'] = self.value
        if self.readonly:
            kwargs['readonly']='readonly'
        return super(MyBS3TextFieldWidget, self).__call__(field, **kwargs)


class MyLineSeparatedListField(Field):
    widget = MyBS3TextAreaFieldWidget()

    # 前端要显示的值
    def _value(self):
        if self.data:
            return u"\n".join(self.data)    # 数据库里面的数据是list
        else:
            return u""

    # 发送到后端的值
    def process_formdata(self, valuelist):
        if valuelist:
            self.data = [x.strip() for x in valuelist[0].split("\n")]
        else:
            self.data = []



class MyJSONField(Field):
    widget = MyBS3TextAreaFieldWidget(rows=3)

    # 前端要显示的值
    def _value(self):
        if self.data:
            # return self.data  #
            if type(self.data)==str:  # 如果是字符集就原样返回
                return self.data
            return json.dumps(self.data,indent=4,ensure_ascii=False)    # 数据库里面的数据是list
        else:
            return u"{}"

    # # 后端发送前端时的数据处理，处理完以后使用_value()进行显示
    # @pysnooper.snoop()
    # def process_data(self, value):
    #     try:
    #         if value:
    #             self.data = json.loads(value)
    #         else:
    #             self.data = {}
    #     except Exception as e:
    #         self.data={}
    #     print(self.data,type(self.data))



    # 发送到后端的值
    def process_formdata(self, valuelist):
        try:
            if valuelist:
                self.data = json.loads(valuelist[0])
            else:
                self.data = {}
        except Exception as e:
            self.data=valuelist[0]  # self.default    # 如果出错，self.data就是原始字符串了。
            raise ValidationError('input must json:'+str(e))


from wtforms.widgets.core import escape_html
from flask_babel import lazy_gettext as _

class MySelect2Widget(object):

    extra_classes = None

    def __init__(self, extra_classes=None, style=None,multiple=False,new_web=True,value='',can_input=False,conten2choices=False):
        self.extra_classes = extra_classes
        self.style = style or u"width:350px"
        self.multiple = multiple
        self.value=value
        self.new_web=new_web
        self.can_input = can_input
        self.conten2choices=conten2choices

    # @pysnooper.snoop()
    def __call__(self, field, **kwargs):
        kwargs["class"] = u"my_select2 form-control"
        if self.extra_classes:
            kwargs["class"] = kwargs["class"] + " " + self.extra_classes
        kwargs["style"] = self.style
        kwargs["data-placeholder"] = _("Select Value")
        if "name_" in kwargs:
            field.name = kwargs["name_"]

        kwargs.setdefault('id', field.id)
        if self.multiple:
            kwargs['multiple'] = True
        if 'required' not in kwargs and 'required' in getattr(field, 'flags', []):
            kwargs['required'] = True
        if self.new_web:
            fun="set_change('%s')"%field.name
        else:
            fun=''

        html = ['''<select %s  id=%s onchange="%s">''' %
                (html_params(name=field.name, **kwargs),field.name,fun)]
        for val, label, selected in field.iter_choices():
            if self.value:
                if str(val)==str(self.value):
                    html.append(self.render_option(val, label, selected=True))
                else:
                    html.append(self.render_option(val, label, selected=False))
            else:
                html.append(self.render_option(val, label, selected))
        html.append('</select>')
        return HTMLString(''.join(html))

    @classmethod
    def render_option(cls, value, label, selected, **kwargs):
        if value is True:
            # Handle the special case of a 'True' value.
            value = text_type(value)

        options = dict(kwargs, value=value)
        if selected:
            options['selected'] = True
        return HTMLString('<option %s>%s</option>' % (html_params(**options), escape_html(label, quote=False)))

# json编辑框
class MyJsonIde(object):
    def __call__(self, field, **kwargs):
        return HTMLString('<pre><code>%s</code></pre>' % (field._value(),))
        # return HTMLString('<pre><code>%s</code></pre>' % (field._value(),))


class MySelect2ManyWidget(widgets.Select):
    extra_classes = None

    def __init__(self, extra_classes=None, style=None,can_input=False):
        self.extra_classes = extra_classes
        self.style = style or u"width:250px"
        self.can_input=can_input
        return super(MySelect2ManyWidget, self).__init__()


from wtforms.fields.core import SelectField
class MySelectMultipleField(SelectField):
    """
    No different from a normal select field, except this one can take (and
    validate) multiple choices.  You'll need to specify the HTML `size`
    attribute to the select field when rendering.
    """
    widget = widgets.Select(multiple=True)

    def iter_choices(self):
        for value, label in self.choices:
            selected = self.data is not None and self.coerce(value) in self.data
            yield (value, label, selected)

    # 将数据库数据处理成前端需要的数据.post的时候也会调用一遍，那时value为None
    # @pysnooper.snoop(watch_explode='value')
    def process_data(self, value):
        try:
            if value:
                self.data = list(self.coerce(v) for v in value.split(','))
                # print(self.data)
            else:
                self.data=None
        except (ValueError, TypeError):
            self.data = None

    # @pysnooper.snoop(watch_explode='valuelist')
    def process_formdata(self, valuelist):
        try:
            self.data = ','.join(list(self.coerce(x) for x in valuelist))
            # print(self.data)
        except ValueError:
            raise ValueError(self.gettext('Invalid choice(s): one or more data inputs could not be coerced'))

    def pre_validate(self, form):
        pass



from flask_appbuilder.widgets import FormWidget
from flask_appbuilder._compat import as_unicode

class MySearchWidget(FormWidget):
    template = "appbuilder/general/widgets/search.html"
    filters = None

    def __init__(self, **kwargs):
        self.filters = kwargs.get("filters")
        self.help_url=kwargs.get("help_url",'')
        return super(MySearchWidget, self).__init__(**kwargs)

    def __call__(self, **kwargs):
        """ create dict labels based on form """
        """ create dict of form widgets """
        """ create dict of possible filters """
        """ create list of active filters """
        label_columns = {}
        form_fields = {}
        search_filters = {}
        dict_filters = self.filters.get_search_filters()
        for col in self.template_args["include_cols"]:
            label_columns[col] = as_unicode(self.template_args["form"][col].label.text)
            form_fields[col] = self.template_args["form"][col]()
            search_filters[col] = [as_unicode(flt.name) for flt in dict_filters[col]]

        kwargs["help_url"] = self.help_url
        kwargs["label_columns"] = label_columns
        kwargs["form_fields"] = form_fields
        kwargs["search_filters"] = search_filters
        kwargs["active_filters"] = self.filters.get_filters_values_tojson()
        return super(MySearchWidget, self).__call__(**kwargs)

