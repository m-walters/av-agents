from collections import namedtuple
from enum import Enum
from typing import Any, Generator, List, Optional, TypedDict, Union

import jax.numpy as jnp
import numpy as np
import pymc
from hydra.utils import instantiate
from omegaconf import DictConfig

# For typing -- It's often hard to say if an object is one or the other
Array = Union[jnp.ndarray, np.ndarray]
# Shortcut
Number = Union[float, int]


def load_param(param_cfg: DictConfig) -> "Param":
    """
    Load a param from a DictConfig
    For instantiating samplers, use the hydra special keys _target_ and _args_, for example
        _target_: pymc.Uniform
        _args_:
            - my_param_name
            - 0
            - 100

    _args_ is needed for positional arguments if needed, but if you know the arg names
    you can just use those directly in the yaml struct.

    See https://hydra.cc/docs/advanced/instantiate_objects/overview/
    """
    if param_cfg.get("sampler", None):
        sampler = instantiate(param_cfg["sampler"])
        return Param(param_cfg.name, sampler=sampler)

    return Param(**param_cfg)


class Param:
    """
    Parameter definition class
    Defines parameters but doesn't perform any sampling itself
    """

    def __init__(
        self,
        name: str,
        sampler: pymc.Distribution | None = None,
        fixed_value: Number | None = None,
    ):
        """
        Must provide one of sampler or fixed_value.
        If fixed value, the fixed_shape must be provided for broadcasting
        """
        if not (isinstance(name, str) and len(name) > 0):
            raise ValueError("Param name must be non-zero string")

        if sampler and fixed_value:
            raise ValueError("Cannot provide both sampler and fixed value")

        self.name = name
        self.sampler = sampler
        self.fixed_value = fixed_value


class ParamCollection:
    """
    Stores a collection of params
    """

    def __init__(self, params: list[Param]) -> None:
        self.collection = {}
        for p in params:
            if hasattr(self, p.name):
                raise ValueError(
                    f"Cannot set attribute on ParamCollection for Param name '{p.name}' since this is reserved"
                )

            self.collection[p.name] = p

            # Init the parameter attribute
            # setattr(self, p.name, None)

        self.tuple_definition = namedtuple('Params', [p.name for p in params])
        # This will be populated after some draws
        self.params = None

        # Declare and set the tuple
        # self.params = namedtuple('Params', [p.name for p in params])()
        # # Add shorthand for retrieving a param
        # setattr(self, self.params.name, p)

    @property
    def samples(self) -> dict:
        """
        Return the sampled values as a parameter dictionary
        """
        return {name: getattr(self.params, name) for name in self.collection}

    def draw(self, draws: int) -> None:
        """
        Draw samples from params in the collection.
        Some may be fixed value
        """
        # Using return_inferencedata=False returns a plain dict
        samples: dict = pymc.sample_prior_predictive(
            draws=draws, return_inferencedata=False
        )

        # Update the collection with the sampled vars
        for p in self.collection.values():
            if p.fixed_value:
                # Broadcast the fixed value
                samples[p.name] = p.fixed_value * jnp.ones((1, draws))

        # Update the tuple
        self.params = self.tuple_definition(**samples)

    def monte_carlo_stack(self, n_montecarlo: int) -> Any:
        """
        Return a tuple vstack of the params for monte carlo simulating
        The returned tuple is like the parent class' params tuple, which is dim (1, draws) for each param,
        but is duplicated across the first dimension, as (n_montecarlo, draws)
        """
        return self.tuple_definition(
            **{
                name: jnp.vstack(
                    [getattr(self.params, name) for _ in range(n_montecarlo)]
                ) for name in self.collection
            }
        )


class ParamEvolution(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    GEOMETRIC = "geometric"
    STEP = "step"


class ParamIteratorConfig(TypedDict):
    evolution: Union[ParamEvolution, str]
    x_0: Number
    x_T: Optional[Number]
    n_steps: Optional[int]


class ParamIterator:
    def __init__(
        self,
        evolution: Union[ParamEvolution, str],
        x_0: Number,
        x_T: Optional[Number] = None,
        n_steps: Optional[int] = None,
        *args, **kwargs
    ):
        self.key = JaxRKey(seed=kwargs.get("seed", 8675309))
        self.value = self.x_0 = x_0
        self.x_T = x_T
        self.n_steps = n_steps

        if isinstance(evolution, str):
            evolution = ParamEvolution(evolution)

        if evolution == ParamEvolution.CONSTANT:
            self.evolver = self.constant_param_evolution(x_0)
        elif evolution == ParamEvolution.LINEAR:
            self.evolver = self.linear_param_evolution(x_0, x_T, n_steps)
        elif evolution == ParamEvolution.EXPONENTIAL:
            self.evolver = self.exponential_param_evolution(x_0, x_T, n_steps)
        elif evolution == ParamEvolution.GEOMETRIC:
            # Alpha is a required argument
            self.evolver = self.geometric_param_evolution(x_0, x_T, n_steps)
        elif evolution == ParamEvolution.STEP:
            # Transitions is a required argument
            self.evolver = self.step_param_evolution(x_0, x_T, n_steps, kwargs["transitions"])
        else:
            raise ValueError(f"Invalid evolution type: '{evolution}'")

    def __call__(self, *args, **kwargs):
        return next(self.evolver)

    def constant_param_evolution(self, x_0: Number) -> Generator:
        while True:
            yield x_0

    def linear_param_evolution(self, x_0: Number, x_end: Number, n_steps: int) -> Generator:
        """
        Yield the next param value in the linear evolution to x_end
        """
        x = x_0
        for i in range(n_steps + 1):
            x = x_0 + i * (x_end - x_0) / n_steps
            self.value = x
            yield x

        # Yield last value for remainder
        while True:
            yield x

    def exponential_param_evolution(self, x_0: Number, x_end: Number, n_steps: int) -> Generator:
        """
        Yield the next param value in the exponential evolution to x_end
        """
        x = x_0
        for i in range(n_steps + 1):
            x = x_0 * (x_end / x_0) ** (i / n_steps)
            self.value = x
            yield x

        # Yield last value for remainder
        while True:
            yield x

    def geometric_param_evolution(self, x_0: Number, x_end: Number, n_steps: int) -> Generator:
        """
        Yield the next param value in geometric-type evolution
        """
        # Let's divide the range into 4 intervals
        if n_steps < 4:
            raise ValueError("Geometric evolution requires at least 4 steps")

        t = n_steps // 4
        t_i = 0
        x = x_0
        for i in range(n_steps + 1):
            if i % t == 0:
                a = 1 / np.power(2, t_i)
                x = a * x_0 + (1 - a) * x_end
                self.value = x
                t_i += 1
            yield x

        # Yield last value for remainder
        while True:
            yield x

    def step_param_evolution(self, x_0: Number, x_end: Number, n_steps: int, transitions: List[int]) -> Generator:
        """
        Yield the next param value in a step evolution.
        The transitions are a list of step indices to transition.
        Transition values are evenly divided based on num transitions and x_end - x_0
        """
        assert len(transitions) > 0
        delta = (x_end - x_0) / len(transitions)
        x = x_0
        for i in range(n_steps + 1):
            if i in transitions:
                x = x + delta
                self.value = x
            yield x

        # Yield last value for remainder
        while True:
            yield x
