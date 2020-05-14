class PyEventBus(object):
    def __init__(self):
        """
        publisher:事件源
        listener:监听者
        event_name:事件id
        """
        self.subscriptions = {}

    def subscribe(self, publisher, event_name, listener, func):
        if listener not in self.subscriptions:
            self.subscriptions[listener] = {}
        if event_name not in self.subscriptions[listener]:
            self.subscriptions[listener][event_name] = {}
        self.subscriptions[listener][event_name][publisher] = func

    def publish(self, publisher, event_name, *args, **kwargs):
        for listener in self.subscriptions.keys():
            if self._has_subscription(publisher, event_name, listener):
                result = self.subscriptions[listener][event_name][publisher](*args, **kwargs)
                if result is not None:
                    return result

    def _has_subscription(self, publisher, event_name, listener):
        return listener in self.subscriptions and \
               event_name in self.subscriptions[listener] and \
               publisher in self.subscriptions[listener][event_name]


class PyPublisher(object):
    def __init__(self):
        """
        publisher:事件源
        listener:监听者
        event_name:事件id
        """
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
