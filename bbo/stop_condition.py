import abc
from abc import ABC


class AlgorithmStopCondition(ABC):
    REASON: str = ""

    @abc.abstractmethod
    def should_stop(self, alg, **kwargs) -> bool:
        raise NotImplementedError()
