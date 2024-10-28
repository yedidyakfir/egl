import logging
import math
from logging import Logger
from typing import List

import numpy as np
import torch
from cma import CMAEvolutionStrategy

from .exception import AlgorithmFinished, NoMoreBudgetError
from .function import Function
from .handlers import AlgorithmCallbackHandler
from .stop_condition import AlgorithmStopCondition
from .trust_region import TrustRegion


class TRCovarianceMatrixAdaptation:
    def __init__(
        self,
        env: Function,
        initial_point: np.ndarray,
        trust_region: TrustRegion = None,
        logger: Logger = None,
    ):
        self.curr_best_point = initial_point
        self.env = env
        self.trust_region = trust_region
        self.logger = logger or logging.getLogger(__name__)

    @property
    def real_cma_evaluator(self):
        def evaluator(x):
            return self.env.evaluate(self.trust_region.inverse(x))

        return evaluator

    def train(
        self,
        num_of_epoch_with_no_improvement: int = None,
        stopping_conditions: List[AlgorithmStopCondition] = None,
        callback_handlers: List[AlgorithmCallbackHandler] = None,
    ):
        self.logger.info(f"starting algorithm cma for space {self.env}")
        callback_handlers = callback_handlers or []
        stopping_conditions = stopping_conditions or []

        for handler in callback_handlers:
            handler.on_algorithm_start(self)

        points_found = np.array([])
        cma_es = CMAEvolutionStrategy(self.curr_best_point, 0.5)
        try:
            no_improvement_counter = 0
            curr_value = math.inf
            while True:
                while not cma_es.stop():
                    self.logger.info(f"CMA - new iteration for {self.env}")
                    solutions = np.array(cma_es.ask())
                    solutions_value = (
                        self.real_cma_evaluator(torch.from_numpy(solutions))
                        .cpu()
                        .numpy()
                    )

                    if len(points_found) > 0:
                        concat_solutions = np.concatenate((points_found, solutions))
                    else:
                        concat_solutions = solutions
                    points_found = np.unique(concat_solutions, axis=0)

                    cma_es.tell(solutions, solutions_value.tolist())
                    cma_es.disp()
                    best_idx = np.argmin(solutions_value)
                    best_point = solutions[best_idx]
                    if solutions_value[best_idx] >= curr_value:
                        self.logger.info(
                            f"No improvement, counter {no_improvement_counter}/{num_of_epoch_with_no_improvement}"
                        )
                        no_improvement_counter += 1
                    else:
                        no_improvement_counter = 0
                        self.logger.info(
                            f"improved by {solutions_value[best_idx] - curr_value}"
                        )
                    self.curr_best_point = best_point
                    curr_value = solutions_value[best_idx]
                    for handler in callback_handlers:
                        handler.on_epoch_end(
                            self,
                            database=torch.tensor(solutions),
                        )

                    for stop_condition in stopping_conditions:
                        if stop_condition.should_stop(
                            self, counter=no_improvement_counter
                        ):
                            raise AlgorithmFinished(
                                stop_condition.REASON.format(
                                    alg="CMA",
                                    env=self.env,
                                    best_point=best_point,
                                    tr=self.trust_region,
                                )
                            )
                real_best_point = self.trust_region.inverse(self.curr_best_point)
                self.logger.info(
                    f"Shrinking CMA around {real_best_point} with {self.trust_region}"
                )
                self.trust_region.squeeze(real_best_point)
                self.curr_best_point = self.trust_region.map(real_best_point)
                cma_es = CMAEvolutionStrategy(self.curr_best_point, 0.5)
                for handler in callback_handlers:
                    handler.on_algorithm_update(self)
        except NoMoreBudgetError as e:
            self.logger.warning("Exceeded budget", exc_info=e)
        except AlgorithmFinished as e:
            self.logger.info(f"CMA Finish stopped {e}")
        finally:
            for handler in callback_handlers:
                handler.on_algorithm_end(self)
            if points_found.size:
                return points_found, self.real_cma_evaluator(
                    torch.from_numpy(points_found)
                )
            return np.array([]), np.array([])
