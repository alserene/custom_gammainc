import pytest
import jax
import jax.numpy as jnp
import numpy as np

from sympy import uppergamma, zoo

from gammainc import custom_gammaincc


@pytest.mark.parametrize("x", [0, 0.1, 0.5, 0.9, 1, 2.1, 3.7, 4.5, 5, 6.6, 7.1, 8.00005, 9.9, 10.0, 15, 18, 20, 25, 30, 100])
@pytest.mark.parametrize("s", [-100, -5, -3.4, -1.2, -1, 0, 0.5, 1, 30, 100, 1000])
def test_inputs(s, x):

    # Get recursion depth from s
    d = np.abs(np.floor(s)).astype(int)

    # Call custom function
    result_jax = custom_gammaincc(s, x, d)
    # Call SymPy function
    result_sympy = uppergamma(s, x).evalf()

    # Excessive print statements to help find the results among the test outputs.
    print("\n\n\n\n\n")
    print("#####     #####     #####     #####     #####     #####     #####     #####")
    print("\n\n")
    print("Comparison against Sympy solutions")
    print("----------------------------------")
    print("\n")
    print(f"Custom Upper Incomplete Gamma (JAX)   s={s}, x={x}:  {result_jax}")
    print(f"Upper incomplete gamma (SymPy)        s={s}, x={x}:  {result_sympy}")
    print("\n\n")   
    print("#####     #####     #####     #####     #####     #####     #####     #####")
    print("\n\n\n\n\n")

    # Compare results
    if result_sympy == zoo: # Complex inf case
        assert jnp.iscomplex(result_jax)
        assert jnp.isinf(result_jax)
    else: # Remaining cases
        assert float(result_jax) == pytest.approx(float(result_sympy))

# The JAX expi function has a bug; this test checks if bug exists.
@pytest.mark.parametrize("x", [0.5])
def test_jax_expi_jit_disabled_bug(x):

    x = jnp.array(x, dtype=jnp.float64)

    jax.config.update("jax_disable_jit", True)
    res_jit_disabled = -jax.scipy.special.expi(-x)
    jax.config.update("jax_disable_jit", False)
    res_jit_enabled = -jax.scipy.special.expi(-x)

    # Excessive print statements to help find the results among the test outputs.
    print("\n\n\n\n\n")
    print("#####     #####     #####     #####     #####     #####     #####     #####")
    print("\n\n")
    print("     git disabled + jax.scipy.special.expi --> bug ")
    print("     ---------------------------------------------")
    print("\n")
    print(f"     jit enabled:      {res_jit_enabled}")
    print(f"     jit disabled:     {res_jit_disabled}")
    print("\n")
    print("      expected result:  0.55977359")
    print("\n\n")   
    print("#####     #####     #####     #####     #####     #####     #####     #####")
    print("\n\n\n\n\n")

    assert res_jit_enabled == pytest.approx(0.55977359)
    assert res_jit_disabled == pytest.approx(0.55977359)

def test_grad():
    """Test that the custom gamma function can be autodifferentiated."""

    jax.config.update("jax_disable_jit", True)

    grad_custom_gammaincc = jax.grad(custom_gammaincc)
    print(grad_custom_gammaincc(-1.2, 2.0, 2))
