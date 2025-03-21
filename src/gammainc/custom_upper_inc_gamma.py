import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import gamma, gammaincc
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import promote_args_inexact

jax.config.update("jax_enable_x64", True)


def s_positive(s: ArrayLike, x: ArrayLike) -> Array:

    # lax.gammaincc is regularised by 1/Γ(s), so we multiply by Γ(s)
    return gamma(s) * (gammaincc(s, x))

def compute_gamma(s: ArrayLike, x: ArrayLike) -> Array:

    def recur(gamma, s, x):
        return lax.div(gamma - jnp.pow(x, s) * lax.exp(-x), s)

    def compute_recurrence(carry, _):
        gamma, s = carry

        # Handle case of JAX inf
        new_gamma = lax.cond(
            jnp.isinf(gamma),
            lambda _: jnp.inf,
            lambda _: recur(gamma, s - 1, x),
            operand=None
        )
        
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

@jit
def custom_gammaincc(s: ArrayLike, x: ArrayLike) -> Array:
    """
    Computes the upper incomplete gamma function Γ(s, x) for real s in range (-3, 3),
    where s != integer, and real x > 0.

    Computation is done by a recurrence relation.

    Parameters:
    s (float): The shape parameter.
    x (float): The lower limit of integration.

    Returns:
    float: The computed value of Γ(s, x).
    """
    
    s, x = promote_args_inexact("custom_gammaincc", s, x)

    return compute_gamma(s, x)
