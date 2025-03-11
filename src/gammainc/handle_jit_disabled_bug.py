

# New version to test which tries to handle the bug and turn jit on and off
def s_zero(x: ArrayLike) -> Array:
    """Special case for s = 0: Gamma(0, x) = -expi^(-x)."""

    # print("S ZERO: ", x)

    # TODO: Is the below correct? e.g. x = jnp.array(2.0)
    # TODO: If jitting, this modification needs to use the jax call.

    def handle_jax_bug(x):

        jax.config.update("jax_disable_jit", True)
        res = -expi(-x)
        jax.config.update("jax_disable_jit", False)

        return res

    def dont_handle_jax_bug(x):

        return -expi(-x)

    # TODO: Type checking review: (inside > out)
    # Ensure x is a JAX double
    x = jnp.array(x, dtype=jnp.float64)

    # Handle x (or not).
    return lax.cond(
        0 <= x <= 1,
        lambda _: handle_jax_bug(x),
        lambda _: dont_handle_jax_bug(x),
        operand=None
    )
