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


class CallableForEpochEnd(AlgorithmCallbackHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_algorithm_start(self, alg, *args, **kwargs):
        pass

    def on_epoch_end(self, alg, *args, **kwargs):
        if self.callback:
            self.callback(alg, *args, **kwargs)

    def on_algorithm_update(self, alg, *args, **kwargs):
        pass

    def on_algorithm_end(self, alg, *args, **kwargs):
        pass
