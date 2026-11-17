class RouterError(Exception):
    pass


class StartupBlocked(RouterError):
    pass


class Busy(RouterError):
    pass


class BackendError(RouterError):
    pass
