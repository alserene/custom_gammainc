import pytest
import jax
import jax.numpy as jnp
import numpy as np

from sympy import uppergamma, zoo

from gammainc import custom_gammaincc


@pytest.mark.parametrize("x", [0.1, 0.5, 0.9, 1, 2.1, 3.7, 4.5, 5, 6.6, 7.1, 8.00005, 9.9, 10.0, 15, 18, 20, 25, 30, 100, 1000])
@pytest.mark.parametrize("s", [-2.95, -1.2, -1.1, 0.02, 0.5, 1.014, 1.7, 2.99])
def test_inputs(s, x):

    # Call custom function
    result_jax = custom_gammaincc(s, x)
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

def test_grad():
    """Test that the custom gamma function can be autodifferentiated."""

    jax.config.update("jax_disable_jit", True)

    grad_custom_gammaincc = jax.grad(custom_gammaincc)
    print(grad_custom_gammaincc(-1.2, 2.0))
