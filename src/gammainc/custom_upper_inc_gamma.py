from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import gamma, gammaincc, expi
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import promote_args_inexact

jax.config.update("jax_enable_x64", True) # NOTE: If comment this out, also update conversion in s_zero(x)
jax.config.update("jax_disable_jit", True)


# @partial(jit, static_argnums=(0,))
def s_zero(x: ArrayLike) -> Array:
    """Special case for s = 0: Gamma(0, x) = e^(-x)."""

    # print("S ZERO: ", x)

    x_val = x
    #TODO: is the below correct? case x = jnp.array(2.0)
    # x_val = jnp.array(x, dtype=jnp.float32)   # Ensure x is a JAX float
    x_val = jnp.array(x, dtype=jnp.float64) # Ensure x is a JAX double

    return -expi(-x_val) # x > 0 (although, seems to work)

# @partial(jit, static_argnums=(0,))
def s_half(x):
    """Solution when s == 1/2."""

    # print(f"S half")

    # Can use any of the following:
    # lax.erfc(x)
    # jax.scipy.special.erfc(x)
    # 1 - jax.scipy.special.erf(x)

    return jnp.sqrt(jnp.pi) * lax.erfc(jnp.sqrt(x))

# @partial(jit, static_argnums=(0,))
def s_one(x):

    # print(f"S one")

    return jnp.exp(-x)

# @partial(jit, static_argnums=(0,1))
def s_positive(s: ArrayLike, x: ArrayLike) -> Array:

    # print(f"S POSITIVE: ", s)

    # print(f"gamma(s): ", gamma(s))
    # print(f"gammaincc(s, x): ", gammaincc(s, x))
    # print(f"gamma(s) * (gammaincc(s, x)) = ", gamma(s) * (gammaincc(s, x)))

    # lax.gammaincc is regularised by 1/Γ(s), so we multiply by Γ(s)
    return gamma(s) * (gammaincc(s, x))

# @partial(jit, static_argnums=(0,1,2))
def s_negative(s: ArrayLike, x: ArrayLike, depth) -> Array:

    # print("S NEGATIVE: ", s)

    def recur(gamma, s, x):
        # print("RECUR variables: ", gamma, s, x)
        # print("RECUR: ", lax.div(gamma - jnp.pow(x, s) * lax.exp(-x), s))
        return lax.div(gamma - jnp.pow(x, s) * lax.exp(-x), s)
        
    def compute_recurrence(carry, _):
        gamma, s = carry

        # Handle case of JAX inf
        if jnp.isinf(gamma):
            new_gamma = jnp.inf
        else:
            new_gamma = recur(gamma, s - 1, x)
        
        # print("new_gamma: ", new_gamma)

        # Return updated state & new gamma
        return (new_gamma, s - 1), new_gamma
    
    # Get recursion starting values
    s_start = s % 1.0
    gamma_start = custom_gammaincc(s_start, x, depth)
    
    # print("gamma_start, s_start: ", gamma_start, s_start)

    # Initiate recursion
    initial_carry = (gamma_start, s_start)
    result, _ = lax.scan(compute_recurrence, initial_carry, None, length=depth)

    # Return final gamma value
    return result[0]

# @partial(jit, static_argnums=(0,1,2))
# def upInGamma()
def custom_gammaincc(s: ArrayLike, x: ArrayLike, depth) -> Array:
    """
    Computes the upper incomplete gamma function Γ(s, x) for any real s and real x >= 0.

    For x = 0
    - Uses the regular gamma function for positive s and non-integer s < 0
    - Returns inf when s = 0 or integer s < 0.
    - Uses regularised lower incomplete gamma function for positive s.
    - Uses recurrence relation for all negative s, including negative integers.
    - Handles the special case of s = 0 correctly using the exponential integral.

    Parameters:
    s (float): The shape parameter.
    x (float): The lower limit of integration.

    Returns:
    float: The computed value of Γ(s, x).
    """
    
    # print(f"s: {s}")
    # print(f"x: {x}")
    # print(f"s.type: {type(s)}")
    # print(f"x.type: {type(x)}")
    s, x = promote_args_inexact("custom_gammaincc", s, x)
    # print(f"s: {s}")
    # print(f"x: {x}")
    # print(f"s.type: {jnp.dtype(s)}")
    # print(f"x.type: {jnp.dtype(s)}")

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

    # # TODO: Make sure each if else returns the same data type
    # return lax.cond(
    #     jnp.isinf(x), # x = inf
    #     lambda _: jnp.array(0, dtype=jnp.float64),
    #     lambda _: lax.cond(
    #         x == 0, # x = 0 cases
    #         lambda _: lax.cond(
    #             s == 0, # s = 0
    #             lambda _: jnp.array(jnp.inf, dtype=jnp.float64),
    #             lambda _: lax.cond(
    #                 jnp.logical_and(s < 0, s == jnp.floor(s)), # s (int) < 0
    #                 # lambda _: lax.complex(jnp.inf, jnp.inf),
    #                 lambda _: jnp.array(jnp.inf, dtype=jnp.float64),
    #                 lambda _: gamma(s), # s > 0 & s (non-int) < 0
    #                 operand=None
    #             ),
    #             operand=None
    #         ),
    #         lambda _: lax.cond( # x > 0 cases
    #             s == 0, # s = 0
    #             lambda _: s_zero(x),
    #             lambda _: lax.cond(
    #                 s == 1/2, # s = 1/2
    #                 lambda _: s_half(x),
    #                 lambda _: lax.cond(
    #                     s == 1, # s = 1
    #                     lambda _: s_one(x),
    #                     lambda _: lax.cond(
    #                         s > 0, # s > 0
    #                         lambda _: s_positive(s, x),
    #                         lambda _: s_negative(s, x, depth), # s < 0
    #                         operand=None
    #                     ),
    #                     operand=None
    #                 ),
    #                 operand=None
    #             ),
    #             operand=None
    #         ),
    #         operand=None
    #     ),
    #     operand=None
    # )
