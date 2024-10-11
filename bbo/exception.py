class AlgorithmError(Exception):
    pass


class AlgorithmFinished(AlgorithmError):
    pass


class NoMoreBudgetError(AlgorithmError):
    pass
