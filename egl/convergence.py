import abc
import copy
import logging
import math
from logging import Logger
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from tqdm.auto import trange

from .exception import NoMoreBudgetError, AlgorithmFinished
from .function import Function
from .handlers import AlgorithmCallbackHandler
from .stop_condition import AlgorithmStopCondition
from .trust_region import TrustRegion
from .utils import distance_between_tensors, ball_perturb
from .value_normalizer import ValueNormalizer


class ConvergenceAlgorithm:
    def __init__(
        self,
        env: Function,
        curr_point: Tensor,
        function_opt: Optimizer,
        epsilon: float,
        epsilon_factor: float,
        min_epsilon: float,
        perturb: float,
        max_batch_size: int = 1024,
        num_of_batch_reply: int = 32,
        maximum_movement_for_shrink: float = math.inf,
        value_normalizer: ValueNormalizer = None,
        trust_region: TrustRegion = None,
        dtype: torch.dtype = torch.float64,
        device: int = None,
        use_tqdm_bar: bool = True,
        logger: Logger = logging.getLogger(__name__),
    ):
        self.env = env
        self.curr_point = curr_point
        self.best_model = copy.deepcopy(curr_point)
        self.function_opt = function_opt
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_factor = epsilon_factor
        self.perturb = perturb
        self.max_batch_size = max_batch_size
        self.num_of_batch_reply = num_of_batch_reply
        self.maximum_movement_for_shrink = maximum_movement_for_shrink
        self.value_normalizer = value_normalizer
        self.trust_region = trust_region
        self.dtype = dtype
        self.device = device
        self.use_tqdm_bar = use_tqdm_bar
        self.logger = logger

    @property
    def best_point_until_now(self):
        return self.best_model.detach().clone()

    @property
    def curr_point_to_draw(self):
        return self.curr_point.detach()

    def real_data(self, data: Tensor):
        if self.trust_region:
            data = self.env.denormalize(self.trust_region.inverse(data))
        return data

    def eval_data(self, data: Tensor):
        data = self.real_data(data)
        return self.env.evaluate(data)

    def train(
        self,
        epochs: int,
        exploration_size: int,
        num_loop_without_improvement: int,
        min_iteration_before_shrink: int,
        surrogate_model_training_epochs: int = 60,
        warmup_minibatch: int = 5,
        warmup_loops: int = 6,
        stopping_conditions: List[AlgorithmStopCondition] = None,
        callback_handlers: List[AlgorithmCallbackHandler] = None,
        **kwargs,
    ):
        self.logger.info(
            f"Starting running {self.__class__.__name__} for {epochs} epochs"
        )
        stopping_conditions = stopping_conditions or []
        callback_handlers = callback_handlers or []

        for callback_handler in callback_handlers:
            callback_handler.on_algorithm_start(self)

        self.training_start_hook(
            epochs,
            exploration_size,
            num_loop_without_improvement,
            min_iteration_before_shrink,
            surrogate_model_training_epochs,
            warmup_minibatch,
            callback_handlers,
            **kwargs,
        )

        # to prevent error if database is not assigned before exception
        database = torch.tensor([])

        try:
            database, evaluations = self.explore(warmup_minibatch * exploration_size)
            num_of_samples = database.shape[-2]
            batch_size = min(self.max_batch_size, num_of_samples)

            self.warm_up(batch_size, database, evaluations, warmup_loops)

            best_model_value = self.eval_data(self.best_point_until_now)
            reply_memory_size = self.num_of_batch_reply * exploration_size
            no_improvement_in_model_count = 0
            counter = 0
            num_of_shrinks = 0
            last_tr_unreal_best = self.best_point_until_now.clone()
            self.logger.info(f"starting on {best_model_value}")

            epoch_loop = (
                trange(epochs, desc=f"Training EGL {epochs} epochs")
                if self.use_tqdm_bar
                else range(epochs)
            )
            for _ in epoch_loop:
                counter += 1
                # Explore
                samples, new_evaluations = self.explore(exploration_size)
                database = torch.cat((database, samples), dim=(len(samples.shape) - 2))[
                    ..., -reply_memory_size:, :
                ]
                evaluations = torch.cat(
                    (evaluations, new_evaluations), dim=(len(evaluations.shape) - 1)
                )[..., -reply_memory_size:]

                num_of_samples = database.shape[-2]
                batch_size = min(self.max_batch_size, num_of_samples)
                test_loss = self.train_surrogate(database, evaluations, batch_size)
                self.train_model()

                # Handle end of epoch
                for handler in callback_handlers:
                    handler.on_epoch_end(
                        self,
                        database=database,
                        test_losses=test_loss,
                        best_model_value=best_model_value,
                    )

                # Check improvement
                new_model_evaluation = self.eval_data(self.curr_point)
                if best_model_value > new_model_evaluation:
                    self.logger.info(
                        f"Improved best known point to {best_model_value} From {new_model_evaluation} In {self.env}"
                    )
                    self.best_model = copy.deepcopy(self.curr_point)
                    best_model_value = new_model_evaluation
                else:
                    self.logger.warning(
                        f"No improvement ({no_improvement_in_model_count}) for value {new_model_evaluation} "
                        f"in {self.env}"
                    )
                    no_improvement_in_model_count += 1

                # Shrink Trust region
                if (
                    no_improvement_in_model_count >= num_loop_without_improvement
                    and counter >= min_iteration_before_shrink
                ):
                    counter = 0
                    no_improvement_in_model_count = 0
                    num_of_shrinks += 1

                    unreal_distance_between_bests = distance_between_tensors(
                        last_tr_unreal_best,
                        self.best_model.detach().clone(),
                    )
                    self.epsilon *= self.epsilon_factor
                    self.epsilon = max(self.epsilon, self.min_epsilon)
                    should_shrink_in_addition_to_move = (
                        self.maximum_movement_for_shrink > unreal_distance_between_bests
                    )
                    if self.trust_region:
                        self.before_shrinking_hook()
                        best_parameters_real = self.trust_region.inverse(
                            self.best_model.detach().clone()
                        )
                        real_database = self.trust_region.inverse(database.detach())
                        if should_shrink_in_addition_to_move:
                            self.logger.info(
                                f"Shrinking trust region, movement {unreal_distance_between_bests}, new center is "
                                f"{self.env.denormalize(best_parameters_real).tolist()} with {self.trust_region}"
                            )

                            self.trust_region.squeeze(best_parameters_real)
                        else:
                            self.trust_region.move_center(best_parameters_real)
                            self.logger.info(
                                f"Moving trust region after moving {unreal_distance_between_bests}, new center is "
                                f"{self.env.denormalize(best_parameters_real).tolist()} with {self.trust_region}"
                            )

                        reply_memory_size = self.num_of_batch_reply * exploration_size
                        self.curr_point = nn.Parameter(
                            self.trust_region.map(best_parameters_real).clone()
                        )
                        self.best_model = copy.deepcopy(self.curr_point)
                        self.after_shrinking_hook()
                        # NOTE - I reset this network only if trust region has changed
                        #       Because If it has not the network should look the same
                        database = self.trust_region.map(real_database)
                        self.warm_up(batch_size, database, evaluations, warmup_loops)
                    self.logger.info(f"Shrinking sample radius to {self.epsilon}")
                    self.logger.info(f"Space status {self.env}")
                    for handler in callback_handlers:
                        handler.on_algorithm_update(
                            self, database=database, best_model_value=best_model_value
                        )

                for stop_condition in stopping_conditions:
                    if stop_condition.should_stop(self, counter=num_of_shrinks):
                        raise AlgorithmFinished(
                            stop_condition.REASON.format(
                                alg=self.__class__.__name__,
                                env=self.env,
                                best_point=self.best_point_until_now,
                                tr=self.trust_region,
                            )
                        )
        except NoMoreBudgetError as e:
            self.logger.warning("No more Budget", exc_info=e)
        except AlgorithmFinished as e:
            self.logger.info(f"{self.__class__.__name__} Finish stopped {e}")

        for handler in callback_handlers:
            self.logger.info(
                f"Calling upon {handler.on_algorithm_end} finishing convergence"
            )
            if database.numel() != 0:
                handler.on_algorithm_end(self, database=database)
            else:
                handler.on_algorithm_end(self)

    def warm_up(self, batch_size, database, evaluations, warmup_loops):
        for i in range(warmup_loops):
            self.logger.info(f"{i} loop for warmup for {self.__class__.__name__}")
            self.train_surrogate(database, evaluations, batch_size)

    def explore(self, exploration_size: int):
        current_model_parameters = self.curr_point.detach()
        new_model_samples = self.samples_points(
            current_model_parameters, exploration_size
        )

        # Evaluate
        evaluations = self.evaluate_point(new_model_samples).to(device=self.device)

        self.value_normalizer.adapt(evaluations)
        return new_model_samples, evaluations

    def samples_points(self, base_point: Tensor, exploration_size: int):
        self.logger.info(
            f"Exploring new data points. Sampling {exploration_size} points"
        )
        new_model_samples = torch.cat(
            (
                ball_perturb(
                    base_point,
                    self.epsilon,
                    exploration_size - 1,
                    self.dtype,
                    self.device,
                ),
                base_point.reshape(1, -1),
            )
        )
        return new_model_samples

    def evaluate_point(self, new_model_samples: Tensor):
        self.logger.info(
            f"Evaluating {len(new_model_samples)} on env with {self.trust_region}"
        )
        return self.eval_data(new_model_samples)

    def train_surrogate(self, samples: Tensor, samples_value: Tensor, batch_size: int):
        raise NotImplementedError()

    def train_model(self):
        raise NotImplementedError()

    def training_start_hook(self, *args, **kwargs):
        pass

    def before_shrinking_hook(self):
        pass

    def after_shrinking_hook(self):
        pass

    @abc.abstractmethod
    def gradient(self, x) -> Tensor:
        raise NotImplementedError()
