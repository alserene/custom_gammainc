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
