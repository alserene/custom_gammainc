from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import gamma, gammaincc, expi
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import promote_args_inexact

jax.config.update("jax_enable_x64", True) # NOTE: If comment this out, also update conversion in s_zero(x)
jax.config.update("jax_disable_jit", True)


def s_zero(x: ArrayLike) -> Array:
    """Solution when s = 0."""

    # TODO: Move typing to custom_gammaincc if possible
    # Ensure x is a JAX double
    x = jnp.array(x, dtype=jnp.float64)

    # TODO: handle JAX bug
    return -expi(-x)

def s_half(x):
    """Solution when s = 1/2."""

    # Can use any of the following:
    # lax.erfc(x)
    # jax.scipy.special.erfc(x)
    # 1 - jax.scipy.special.erf(x)

    return jnp.sqrt(jnp.pi) * lax.erfc(jnp.sqrt(x))

def s_one(x):
    """Solution when s = 1."""

    return jnp.exp(-x)

def s_positive(s: ArrayLike, x: ArrayLike) -> Array:

    # lax.gammaincc is regularised by 1/Γ(s), so we multiply by Γ(s)
    return gamma(s) * (gammaincc(s, x))

def s_negative(s: ArrayLike, x: ArrayLike, depth) -> Array:

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
    s_start = s % 1.0
    gamma_start = custom_gammaincc(s_start, x, depth)
    
    # Initiate recursion
    initial_carry = (gamma_start, s_start)
    result, _ = lax.scan(compute_recurrence, initial_carry, None, length=depth)

    # Return final gamma value
    return result[0]

def custom_gammaincc(s: ArrayLike, x: ArrayLike, depth) -> Array:
    """
    Computes the upper incomplete gamma function Γ(s, x) for any real s and real x >= 0.

    For x = inf
                        - Returns 0.

    For x = 0
    - [s = 0]           - Returns inf
    - [s (int) < 0]     - Returns commplex inf.
    - [s (non-int) < 0] - Returns gamma(x).
    - [s > 0]           - Returns gamma(x).

    For x > 0
    - [s = 0]           - Returns -expi(-x)
    - [s = 0.5]         - Returns jnp.sqrt(jnp.pi) * lax.erfc(jnp.sqrt(x))
    - [s = 1]           - Returns jnp.exp(-x)
    - [s > 0]           - Uses regularised lower incomplete gamma function.
    - [s < 0]           - Uses recurrence relation for all negative s, including negative integers.

    Parameters:
    s (float): The shape parameter.
    x (float): The lower limit of integration.
    d (int):   The depth of recursion: np.abs(np.floor(s)).astype(int)

    Returns:
    float: The computed value of Γ(s, x).
    """
    
    s, x = promote_args_inexact("custom_gammaincc", s, x)

    return lax.cond(
        jnp.isinf(x), # x = inf
        lambda _: 0,
        lambda _: lax.cond(
            x == 0, # x = 0 cases
            lambda _: lax.cond(
                s == 0, # s = 0
                lambda _: jnp.inf,
                lambda _: lax.cond(
                    jnp.logical_and(s < 0, s == jnp.floor(s)), # s (int) < 0
                    lambda _: lax.complex(jnp.inf, jnp.inf),
                    lambda _: gamma(s), # s > 0 & s (non-int) < 0
                    operand=None
                ),
                operand=None
            ),
            lambda _: lax.cond( # x > 0 cases
                s == 0, # s = 0
                lambda _: s_zero(x),
                lambda _: lax.cond(
                    s == 1/2, # s = 1/2
                    lambda _: s_half(x),
                    lambda _: lax.cond(
                        s == 1, # s = 1
                        lambda _: s_one(x),
                        lambda _: lax.cond(
                            s > 0, # s > 0
                            lambda _: s_positive(s, x),
                            lambda _: s_negative(s, x, depth), # s < 0
                            operand=None
                        ),
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            ),
            operand=None
        ),
        operand=None
    )
