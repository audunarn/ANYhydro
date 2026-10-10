# Oscillatory radiation quadrature

For long kernels, the legacy frequency-node trapezoid computes a sum of cosines.
On a uniform frequency grid it repeats at 2*pi/delta_omega. Increasing the time
horizon cannot remove that recurrence. The existing default and serialized
artifacts retain their arithmetic and hashes.

An explicit `RadiationSettings(quadrature_method="piecewise_linear_exact", ...)`
uses the exact cosine integral of the componentwise linear damping interpolant
over the same declared finite band, including the selected low-frequency tail.
The high-frequency tail remains truncated; no coefficients are extrapolated,
symmetrized or clamped. The method is serialized, hashed and saved/reopened.

For interval midpoint m and halfwidth h, endpoint average M and half-difference
D, z=h*t, its integral is

`2*h*(M*cos(m*t)*sinc(z) + D*sin(m*t)*sinc_prime(z))`.

Here sinc(z)=sin(z)/z; the derivative uses its analytic small-z series. At t=0,
the interval integral is `(b-a)*(Ba+Bb)/2`. Time blocks bound temporary storage.
The common numerical core serves both neutral-dataset and derived-operator
builders. Explicit time windows are applied after integration in both paths.

This repairs oscillatory quadrature error for the **declared interpolant**. It
does not establish BEM interpolation accuracy, band/tail adequacy, A_inf accuracy,
finite-time or discrete passivity, RAO/free-decay acceptance or qualification.
Both added-mass and damping reconstruction and the existing independent
scientific gates still apply. The legacy duration/spacing diagnostic remains a
spacing indicator; it is not an alias recurrence criterion for the new method.
Keep raw source samples and historical failed artifacts.

Evidence and independent assessment:
`C:/ANYworkspaceWork/anymarine-evidence/w5-oscillatory-20261010/`.
