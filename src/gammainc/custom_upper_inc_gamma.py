from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import gamma, gammaincc, expi
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import promote_args_inexact

jax.config.update("jax_enable_x64", True) # NOTE: If comment this out, also update conversion in s_zero(x)
jax.config.update("jax_disable_jit", True)


def s_positive(s: ArrayLike, x: ArrayLike) -> Array:

    # lax.gammaincc is regularised by 1/Γ(s), so we multiply by Γ(s)
    return gamma(s) * (gammaincc(s, x))

def s_negative(s: ArrayLike, x: ArrayLike) -> Array:

    def recur(gamma, s, x):
        return lax.div(gamma - jnp.pow(x, s) * lax.exp(-x), s)
        
    def compute_recurrence(carry, _):
        gamma, s = carry

        # Handle case of JAX inf
        if jnp.isinf(gamma):
            new_gamma = jnp.inf
        else:
            new_gamma = recur(gamma, s - 1, x)
        
        # Return updated state & new gamma
        return (new_gamma, s - 1), new_gamma
    
    # Get recursion starting values
    recur_depth = 3
    s_start = s + 3
    gamma_start = s_positive(s_start, x)
    
    # Initiate recursion
    initial_carry = (gamma_start, s_start)
    result, _ = lax.scan(compute_recurrence, initial_carry, None, length=recur_depth)

    # Return final gamma value
    return result[0]

def custom_gammaincc(s: ArrayLike, x: ArrayLike) -> Array:
    """
    Computes the upper incomplete gamma function Γ(s, x) for any real s and real x >= 0.

    For x = inf
    - Returns 0.

    For x = 0
    - [s = 0]           - Return inf
    - [s (int) < 0]     - Returns commplex inf.
    - [s (non-int) < 0] - Returns gamma(x).
    - [s > 0]           - Returns gamma(x).

    For x > 0
    - [s = 0] - Returns -expi(-x)
    - [s = 0.5] - jnp.sqrt(jnp.pi) * lax.erfc(jnp.sqrt(x))
    - [s = 1] - Uses regularised lower incomplete gamma function for positive s.
    - [s > 0] - Uses recurrence relation for all negative s, including negative integers.
    - [s < 0] - Handles the special case of s = 0 correctly using the exponential integral.

    Parameters:
    s (float): The shape parameter.
    x (float): The lower limit of integration.

    Returns:
    float: The computed value of Γ(s, x).
    """
    
    s, x = promote_args_inexact("custom_gammaincc", s, x)

    return s_negative(s, x)
