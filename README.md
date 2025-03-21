# custom_gammainc

$$
\Gamma(s, x) = \int_x^\infty t^{s-1} e^{-t} \, dt
$$

A custom JAX implementation of the **upper incomplete gamma function**, <code>Γ(s, x)</code>, for real <code>s</code> in range <code>(-3, 3)</code>, where <code>s != integer</code>, and <code>real x > 0</code>.

### Purpose

- This function is useful for testing the cosmological principle (that the universe is uniformly isotropic and homogeneous).
- This is proposed to be done using Bayesian hierarchical modelling to measure variations in the density of galaxies across the sky and with redshift.

### Method

- <code>Γ(s, x)</code> is computed using the following recursion relation:

$$
\Gamma(s+1, x) = s \Gamma(s, x) + x^s e^{-x}
$$

### Details

Parameters:
 - <code>s (float)</code>: The shape parameter.
 - <code>x (float)</code>: The lower limit of integration.

Returns:
 - <code>float</code>: The computed value of <code>Γ(s, x)</code>.


# Instructions

### Package/dependency Manager
To get the notebook running with custom_gammainc, I recommend you to use [Poetry](https://python-poetry.org/). Instructions for use are below. If you need help installing, let me know :)

### Setup
1. Clone code from GitHub
```
git clone https://github.com/alserene/custom_gammainc.git
```
2. Set up Poetry
```
cd custom_gammainc
poetry install
```
3. Checkout [recur_only](https://github.com/alserene/custom_gammainc/tree/recur_only) branch
```
git checkout recur_only
```
4. Build project
```
poetry build
```
6. Run [tests](https://github.com/alserene/custom_gammainc/blob/recur_only/tests/test_upper_inc_gamma.py) to check all is as expected
```
poetry run pytest tests/test_upper_inc_gamma.py
```

### Notebook
To run the notebook, [schechter_fit_numpyro_jax.ipynb](https://github.com/alserene/custom_gammainc/blob/recur_only/src/gammainc/schechter_fit_numpyro_jax.ipynb), you will need to add data to the same folder or modify the data path (defintion at top of notebook).

# Other
If you want to add packages to poetry env:

<code>poetry add \<package> </code>
