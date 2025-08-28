class TimeoutException(BaseException):
    pass

def raise_timeout(*args):
    raise TimeoutException