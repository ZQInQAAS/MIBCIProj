import threading


class PyPublisher(object):
    def __init__(self):
        # publisher:事件源; listener:监听者; event_name:事件id
        super(PyPublisher, self).__init__()
        self.subscriptions = {}

    def subscribe(self, event_name, func):
        if event_name not in self.subscriptions:
            self.subscriptions[event_name] = []
        self.subscriptions[event_name].append(func)

    def publish(self, event_name, *args, **kwargs):
        if self._has_subscription(event_name):
            for func in self.subscriptions[event_name]:
                result = func(*args, **kwargs)
                if result is not None:
                    return result

    def _has_subscription(self, event_name):
        return event_name in self.subscriptions


class RepeatingTimer(threading.Timer):
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)


# LazyProperty 虚拟代理 惰性地（首次使用时）创建对象 修饰器
class LazyProperty:
    def __init__(self, method):
        self.method = method
        self.method_name = method.__name__
        # print('function overriden: {}'.format(self.fget))
        # print("function's name: {}".format(self.func_name))

    def __get__(self, obj, cls):
        if not obj:
            return None
        value = self.method(obj)
        # print('value {}'.format(value))
        setattr(obj, self.method_name, value)
        return value
