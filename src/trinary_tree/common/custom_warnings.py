import warnings


class MissingFeatureWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class ExtraFeatureWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
