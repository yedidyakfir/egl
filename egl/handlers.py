from abc import ABC


class AlgorithmCallbackHandler(ABC):
    def on_algorithm_start(self, alg, *args, **kwargs):
        raise NotImplementedError()

    def on_epoch_end(self, alg, *args, **kwargs):
        raise NotImplementedError()

    def on_algorithm_update(self, alg, *args, **kwargs):
        raise NotImplementedError()

    def on_algorithm_end(self, alg, *args, **kwargs):
        raise NotImplementedError()
