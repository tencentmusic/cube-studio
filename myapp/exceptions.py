class MyappException(Exception):
    status = 500

    def __init__(self, msg):
        super(MyappException, self).__init__(msg)


class MyappTimeoutException(MyappException):
    pass

class QueryClauseValidationException(MyappException):
    status = 400

class MyappSecurityException(MyappException):
    status = 401

    def __init__(self, msg, link=None):
        super(MyappSecurityException, self).__init__(msg)
        self.link = link


class MetricPermException(MyappException):
    pass


class NoDataException(MyappException):
    status = 400


class NullValueException(MyappException):
    status = 400


class MyappTemplateException(MyappException):
    pass


class SpatialException(MyappException):
    pass


class DatabaseNotFound(MyappException):
    status = 400
