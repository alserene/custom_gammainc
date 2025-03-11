import pytest
import jax
import jax.numpy as jnp
import numpy as np
from sympy import uppergamma, zoo, oo

import gammainc
from gammainc import custom_gammaincc

# Test sets --------------------------------------------------------------------------------------

# --- Full test set
@pytest.mark.parametrize("x", [0, 0.1, 0.5, 0.9, 1, 2.1, 3.7, 4.5, 5, 6.6, 7.1, 8.00005, 9.9, 10.0, 15, 18, 20, 25, 30, 100, 1000])
# Conditionally passing
# pass when comment out x_val = jnp.array(x, dtype=jnp.float64), wrong value otherwise
# @pytest.mark.parametrize("x", [0, 0.1, 0.5, 0.9, 1])
# pass when x_val = jnp.array(x, dtype=jnp.float64)
# @pytest.mark.parametrize("x", [2.1, 3.7, 4.5, 5, 6.6, 7.1, 8.00005, 9.9, 10.0, 15, 18, 20, 25, 30, 100, 1000])

# --- Full test set
@pytest.mark.parametrize("s", [-1000, -100, -5, -3.4, -1.2, -1, 0, 0.5, 1, 30, 100, 1000, 10000])


# Failures ---------------------------------------------------------------------------------------

# FAILS related to x_val = jnp.array(x, dtype=jnp.float64) in s_zero(x)
#   0<=x<=1 fails for s<=0 when x_val = jnp.array(x, dtype=jnp.float64) in s_zero(x), pass when commented out
#   x>=2 fails for s<=0, only passes when x_val = jnp.array(x, dtype=jnp.float64) is NOT commented out

# Only other test not passing (not considering type errors as described above)
# @pytest.mark.parametrize("x", [1000])
# @pytest.mark.parametrize("s", [-1000])
# -> sympy fails to converge, custom jax gives 0.0

# @pytest.mark.parametrize("x", [0.5])
# @pytest.mark.parametrize("s", [0])
def test_inputs(s, x):

    # print(f"s.type: {type(s)}")
    # print(f"x.type: {type(x)}")

    # Get recursion depth from s
    d = np.abs(np.floor(s)).astype(int)

    # Call custom function
    result_jax = custom_gammaincc(s, x, d)
    # Call SymPy function
    result_sympy = uppergamma(s, x).evalf()

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
    
@pytest.mark.parametrize("x", [0.5])
def test_jax_expi_jit_disabled_bug(x):

    x = jnp.array(x, dtype=jnp.float64)

    jax.config.update("jax_disable_jit", True)
    res_jit_disabled = -jax.scipy.special.expi(-x)
    jax.config.update("jax_disable_jit", False)
    res_jit_enabled = -jax.scipy.special.expi(-x)

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

    # assert 1 == 2
