import json
from typing import Union

import numpy as np
import xarray as xr

# For typing -- It's often hard to say if an object is one or the other
# Array = Union[jnp.ndarray, np.ndarray]
Array = Union[np.ndarray]

# Shortcut
Number = Union[float, int]


####
# Deprecated for now
#
# class JaxRKey:
#     """
#     Helper class for seeding RNG with Jax
#     """
#
#     def __init__(self, seed):
#         self.key = jrandom.PRNGKey(seed)
#
#     def next_key(self):
#         # Use subkey to seed your functions
#         self.key, subkey = jrandom.split(self.key)
#         return subkey
#
#     def next_seed(self):
#         # When you want the next int and not next j-key tuple
#         return int(self.next_key()[0])
#
#     def next_seeds(self, n):
#         self.key, *subkeys = jrandom.split(self.key, n + 1)
#         return [int(k[0]) for k in subkeys]


class NpyRKey:
    """
    Helper class for seeding RNG with numpy
    """

    def __init__(self, seed):
        # Initialize the random generator with the given seed
        self.rng = np.random.default_rng(seed)

    def next_key(self):
        raise NotImplementedError("This method is for the JaxRKey class")

    def next_seed(self):
        # Return a single random integer seed
        return int(self.rng.integers(0, 2 ** 32))

    def next_seeds(self, n):
        # Generate n random integers in a single call
        return list(map(int, self.rng.integers(0, 2 ** 32, size=n).tolist()))


# class JaxGaussian:
#     @staticmethod
#     def log_prob(point, loc, scale):
#         var = scale ** 2
#         denom = jnp.sqrt(2 * jnp.pi * var)
#         log_probs = -0.5 * ((point - loc) ** 2) / var - jnp.log(denom)
#         return log_probs
#
#     @staticmethod
#     def sample(key, loc, scale):
#         sample = jax.random.normal(key, loc.shape) * scale + loc
#         log_probs = JaxGaussian.log_prob(sample, loc=loc, scale=scale)
#         return sample, log_probs
#


def combine_results(ds1: xr.Dataset, ds2: xr.Dataset, dim: str) -> xr.Dataset:
    """
    Combine along dimension dim.
    Shifts dimension of ds2 up by the max in ds1
    """
    if dim != "world":
        raise NotImplementedError(f"combine_results only configured for 'world' dimension")

    if dim == "world":
        ds2 = ds2.assign_coords(world=ds2.world + int(ds1.world[-1]) + 1)
        return xr.concat([ds1, ds2], dim=dim)


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
