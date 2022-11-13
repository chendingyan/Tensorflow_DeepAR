class DataSchemaCheckException(Exception):
    def __init__(self, e):
        self.exception = e

    def __repr__(self):
        return self.exception

class ParameterException(Exception):
    def __init__(self, e):
        self.exception = e

    def __repr__(self):
        return self.exception