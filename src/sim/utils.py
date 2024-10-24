import json
from collections import namedtuple
from enum import Enum
from typing import Generator, List, Optional, TypedDict, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import xarray as xr

Params = namedtuple('Params', 'B, w, r, k, qE')
FishParams = namedtuple('Params', 'B, r, k, qE')  # Remove 'w'

# For typing -- It's often hard to say if an object is one or the other
Array = Union[jnp.ndarray, np.ndarray]
# Shortcut
Number = Union[float, int]


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


class JaxRKey:
    """
    Helper class for seeding RNG with Jax
    """

    def __init__(self, seed):
        self.key = jrandom.PRNGKey(seed)

    def next_key(self):
        # Use subkey to seed your functions
        self.key, subkey = jrandom.split(self.key)
        return subkey

    def next_seed(self):
        # When you want the next int and not next j-key tuple
        return int(self.next_key()[0])

    def next_seeds(self, n):
        self.key, *subkeys = jrandom.split(self.key, n + 1)
        return [int(k[0]) for k in subkeys]


class JaxGaussian:
    @staticmethod
    def log_prob(point, loc, scale):
        var = scale ** 2
        denom = jnp.sqrt(2 * jnp.pi * var)
        log_probs = -0.5 * ((point - loc) ** 2) / var - jnp.log(denom)
        return log_probs

    @staticmethod
    def sample(key, loc, scale):
        sample = jax.random.normal(key, loc.shape) * scale + loc
        log_probs = JaxGaussian.log_prob(sample, loc=loc, scale=scale)
        return sample, log_probs


class Output:
    def __init__(
        self,
        Es: List[Array],
        Bs: List[Array],
        Vs: List[Array],
        Rts: List[Array],
        Lts: List[Array],
        param_history: dict[str, list[Number]] | None = None,
    ):
        """
        Outputs from the main world model sim.
        Each variable is a list of length duration,
        and each element therein is an array of shape either (1, world_draws) or (world_draws).
        The stored values become numpy ndarrays of shape [duration, world_draws].

        param_history
        """
        self.Es = np.stack(Es).squeeze()
        self.Bs = np.stack(Bs).squeeze()
        self.Vs = np.stack(Vs).squeeze()
        self.Rts = np.stack(Rts).squeeze()
        self.Lts = np.stack(Lts).squeeze()
        self.param_history = {}
        if param_history is not None:
            # numpify
            for p, vals in param_history.items():
                # Validate that the shapes are right and tight, and don't have a bad name
                assert self.Es.shape[0] == len(vals)
                assert p not in ["E", "B", "V", "Rt", "Lt"]
                self.param_history[p] = np.array(vals)

        assert self.Es.shape == self.Bs.shape == self.Vs.shape == self.Rts.shape == self.Lts.shape

    def plot(self):
        pass


class Results:
    @staticmethod
    def save_ds(ds: xr.Dataset, path):
        """
        Save an omega dataset to disk
        """
        ds.to_netcdf(path)

    @staticmethod
    def load_ds(path):
        """
        Load a saved dataset
        """
        return xr.open_dataset(path)

    @staticmethod
    def save_json(data, path):
        """
        Save a dict to a json file
        """
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load_json(path):
        """
        Load a json file
        """
        with open(path, "r") as f:
            return json.load(f)


class OmegaResults(Results):
    """
    Class for saving runs that iterate over omega
    Outputs must have dimensions/coordinates [omega, time, param_batch]
    """

    def __init__(
        self,
        omegas: Union[np.ndarray, List[float]],
        outputs: List[Output],
        duration: int,
        world_draws: int,
    ):
        self.omegas = omegas
        self.outputs = outputs
        self.duration = duration
        self.world_draws = world_draws

    def to_dataset(self) -> xr.Dataset:
        """
        For runs with varying omega, generate the xArray Dataset
        Outputs must have dimensions [omega, time, param_batch]
        """
        Es = [output.Es for output in self.outputs]
        Bs = [output.Bs for output in self.outputs]
        Vs = [output.Vs for output in self.outputs]
        Rts = [output.Rts for output in self.outputs]

        ds = xr.Dataset(
            {
                "E": (("omega", "time", "batch"), Es),
                "B": (("omega", "time", "batch"), Bs),
                "V": (("omega", "time", "batch"), Vs),
                "Rt": (("omega", "time", "batch"), Rts),
            },
            coords={
                "omega": self.omegas,
                "time": np.arange(self.duration),
                "batch": np.arange(self.world_draws),
            },
        )
        return ds


class ProjectionResults(Results):
    """
    For risk projection analysis
    Outputs must have dimensions/coordinates [Es, time, param_batch]
    """

    def __init__(
        self,
        Es: Union[np.ndarray, List[float]],
        outputs: List[Output],
        duration: int,
        world_draws: int,
    ):
        self.Es = Es
        self.outputs = outputs
        self.duration = duration
        self.world_draws = world_draws

    def to_dataset(self) -> xr.Dataset:
        """
        Outputs must have dimensions [qEs, time, param_batch]
        """
        Bs = [output.Bs for output in self.outputs]
        Vs = [output.Vs for output in self.outputs]
        Rts = [output.Rts for output in self.outputs]

        ds = xr.Dataset(
            {
                "B": (("E", "time", "batch"), Bs),
                "V": (("E", "time", "batch"), Vs),
                "Rt": (("E", "time", "batch"), Rts),
            },
            coords={
                "E": self.Es,
                "time": np.arange(self.duration),
                "batch": np.arange(self.world_draws),
            },
        )
        return ds


class EvolvePreferenceResults(Results):
    """
    For the preference evolution results
    """

    def __init__(
        self,
        output: Output,
        duration: int,
        world_draws: int,
    ):
        self.output = output
        self.duration = duration
        self.world_draws = world_draws

    def to_dataset(self) -> xr.Dataset:
        """
        Outputs must have dimensions [time, param_batch]
        """
        Es = self.output.Es
        Bs = self.output.Bs
        Vs = self.output.Vs
        Rts = self.output.Rts
        Lts = self.output.Lts

        # Params don't have a batch dimension since they are constant across num_param_batch
        param_data = {
            p: (("time",), self.output.param_history[p])
            for p in self.output.param_history
        }

        ds = xr.Dataset(
            {
                "E": (("time", "batch"), Es),
                "B": (("time", "batch"), Bs),
                "V": (("time", "batch"), Vs),
                "Rt": (("time", "batch"), Rts),
                "Lt": (("time", "batch"), Lts),
                **param_data,
            },
            coords={
                "time": np.arange(self.duration),
                "batch": np.arange(self.world_draws),
            },
        )
        return ds
