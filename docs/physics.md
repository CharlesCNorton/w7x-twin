# The physics record

Topic accounts of what the twin computes and how each number was checked. Every table
here is a snapshot of a record under `results/`, which is the current version of it;
the commands named in each section regenerate both.

## Transport

`transport.py` solves the temperature profiles from a steady-state one-dimensional
power balance on the equilibrium's own flux geometry.
Power crossing each surface is the heating deposited inside it, and a diffusive
closure converts that flux into a gradient, integrated inward from the separatrix.

The diffusivity's magnitude is solved for by requiring the resulting global energy
confinement time to equal the ISS04 international stellarator scaling at the same minor
and major radius, density, field and heating power. The scaling supplies the global
confinement, the power balance supplies the profile shape, and the equilibrium supplies
the geometry.

| Heating | τ_E (ISS04) | W | T_e(0) | T_i(0) | χ(0) | χ(a) |
|---|---|---|---|---|---|---|
| 2 MW | 0.296 s | 0.59 MJ | 2.18 keV | 1.20 keV | 0.17 m²/s | 0.81 m²/s |
| 5 MW | 0.169 s | 0.85 MJ | 3.20 keV | 1.76 keV | 0.28 m²/s | 1.35 m²/s |
| 10 MW | 0.111 s | 1.11 MJ | 4.25 keV | 2.34 keV | 0.42 m²/s | 2.02 m²/s |

The table is single-channel, one diffusivity carrying all the transport. Supplying a
neoclassical diffusivity holds that part at its computed value and scales only the
remainder, which changes the solution; the Neoclassical transport section carries that
case.

## Where the heating is absorbed

`transport.py` puts the power where the field and the profiles decide rather than on a
prescribed Gaussian. The electron-cyclotron resonance is a surface in |B|: at 140 GHz the
second harmonic in X-mode resonates at 2.5007 T, and the standard configuration at 12883 A per
turn carries 2.3600 T on axis with |B| spanning 2.2653 to 2.5392 T across the plasma, so the
resonance lies inside 19 of 81 surfaces and the layer weighting puts its absorption at
s = 0.675.

The traced ray corrects that placement. `transport.trace_cyclotron_ray` marches the launched
beam by the scalar eikonal at the local Appleton-Hartree index on the extraordinary branch,
bending through the density it crosses, with the group-velocity anisotropy of the magnetized
branch the stated approximation. Launched from the outer midplane at the bean plane and
aimed at the axis, the ray crosses the resonance at s = 0.020 after 0.71 m of path: the
bean-plane axis field sits just above the resonant 2.5007 T, so central aiming deposits
centrally, which the per-surface layer weighting cannot see because it scales every surface
from the averaged axis field. The deposition it leaves is 0.014 wide in s about the
crossing, floored by the beam waist mapped through the local flux gradient.

The R-wave cutoff at the h-th harmonic leaves n_e = (1 - 1/h) w^2 e0 m_e / e^2, which for X2 at
140 GHz is 1.216 x 10^20 m^-3. A discharge past it has to be heated in O-mode, which is what
the pellet programmes did to reach 10^20 m^-3.

Beam power is attenuated along its own path by the line-integrated density, absorbing 67 % of a
55 keV beam with the deposition peaking at s = 0.250. The fast ions then slow down on both
species, and the critical energy runs 103.6 keV on axis to 3.0 keV at the edge, so 87.5 % of
the beam power reaches the ions on axis and 10.5 % at the edge.

## Stability beyond the Mercier criterion

`diagnostics.py` adds the ballooning limit and the tearing index to the interchange criterion
VMEC returns. `python -m w7x_twin stability` runs all three against beta:

| ⟨β⟩ | Mercier unstable | Ballooning unstable | Max α | Rationals crossed |
|---|---|---|---|---|
| 0.50 % | 0.102 | 0.000 | 0.327 | 0 |
| 1.05 % | 0.061 | 0.314 | 0.694 | 0 |
| 2.00 % | 0.061 | 0.706 | 1.367 | 0 |
| 2.99 % | 0.061 | 0.843 | 2.163 | 0 |

The ballooning drive goes as the pressure gradient and the shear that holds it does not move
with beta, so the unstable fraction runs from nothing at half a percent to 0.843 at three. The
transform spans 0.85 to 0.96 and crosses no rational of order six or below, which is why the
island chain the divertor uses sits outside the last closed surface rather than inside it, and
why the tearing index has no resonance to be evaluated at.

## A discharge through its waveform

`current.py` advances the stored energy and the net current through a heating trace instead of
solving two steady states. The two time constants are far apart: at 1750 eV the plasma's
inductive time is 28.4 s against an energy confinement of 0.162 s.

`python -m w7x_twin history` runs 20180919.033, two megawatts of electron-cyclotron heating
replaced at 1.7 s by 3.4 MW of beam:

| t | P | W | I | ι at edge |
|---|---|---|---|---|
| 1.00 s | 2.0 MW | 0.443 MJ | −0.14 kA | 0.87857 |
| 1.60 s | 2.0 MW | 0.448 MJ | −0.25 kA | 0.87873 |
| 1.80 s | 3.4 MW | 0.502 MJ | −0.28 kA | 0.87878 |
| 2.50 s | 3.4 MW | 0.551 MJ | −0.43 kA | 0.87900 |
| 6.00 s | 3.4 MW | 0.552 MJ | −1.13 kA | 0.88004 |

The energy settles 0.29 s after the step and the current takes 4.01 s to come within five per
cent of where it is going, which it never reaches: at 6 s it carries 1.13 kA of the 6.5 kA the
stored energy implies. The measured current of that scan reached about 5 kA after six seconds,
so the model is low by a factor of four while showing the same behaviour of a current still
rising when the discharge ends. The beam-phase stored energy is 0.552 MJ against a measured
0.50, where the steady-state solve at the record confinement gave 1.02.

### One transient solution, the layer closing the edge

The trace above advances two scalars through fitted closures. `python -m w7x_twin
transient` marches the profiles themselves: the density and the temperature evolve
together on the same radial grid, by the backward-Euler conservation laws
`transport.evolve_density` and `transport.temperature_step`, and at every step the
scrape-off layer is closed at that instant's crossing power and edge density. The
upstream temperature the two-point model returns is the pedestal the core profile
stands on, 111 eV at 2 MW and 156 at 3.4 MW against the 100 the stationary solve holds,
so the edge boundary condition is the layer's answer rather than a held number. The
diffusivity is each phase's stationary closure, so the march carries the dynamics
between the two rather than a confinement closure of its own, and the density starts on
the particle closure's own steady state at the discharge's axis density, which the
source is scaled to hold.

From a cold start the flat-top holds 0.333 MJ against the stationary balance's 0.311 at
the same power and closure, the difference being the pedestal the layer sets and the
cylindrical metric the march conserves against the balance's flux geometry. After the
power step the energy settles within five per cent in 0.14 s, and at six seconds the
current has reached 0.9 kA of the 5.0 the energy implies, on a 28.6 s inductive time: a
current still rising when the discharge ends, which is what the machine's own measured
5 kA at six seconds also shows. Record `results/plasma/transient_discharge.json`.

## Perturbations that break the field periodicity

VMEC labels toroidal harmonics in units of the field period, so an equilibrium at
`nfp = 5` can only carry multiples of five and a single energised trim coil cannot
enter it. `field.py` rewrites a periodic input over the whole torus at `nfp = 1`,
where the five-period structure is carried by n = 5, 10, 15 and everything between is
available to a perturbation. The rewriting is exact: each mode n becomes n·nfp.

With the trim circuits unpowered the solution recovers the five-fold periodicity
without being asked to, leaving n = 1 to 4 at 10⁻¹¹ m, and reproduces the periodic
solve to five digits. Energising `trim_a1` at 1800 A/turn populates them:

| Boundary harmonic | Trim off | trim_a1 at 1800 A |
|---|---|---|
| n = 1 | 1.7 × 10⁻¹¹ m | 4.04 × 10⁻¹ m |
| n = 2 | 1.6 × 10⁻¹¹ m | 6.91 × 10⁻² m |
| n = 3 | 9.9 × 10⁻¹² m | 5.14 × 10⁻² m |
| n = 4 | 1.8 × 10⁻¹¹ m | 1.75 × 10⁻¹ m |
| n = 5 | 1.010 m | 0.977 m |
| Edge transform | 0.95333 | 0.98969 |

The amplitude of that response follows from the reconstructed trim coil geometry, so
A Sobol scan over the reconstruction bounds it. The
published dimensions fix the widths, heights and turn counts; the mounting radius, the
corner rounding and the placement within a module are inferred. Varying all seven
inferences on a Sobol sequence and measuring the harmonics of B_r on the R = 6.2 m
midplane circle at 1800 A/turn:

| n | Nominal | Median | 5th | 95th | Spread |
|---|---|---|---|---|---|
| 1 | 1.479 mT | 1.469 mT | 1.173 mT | 1.870 mT | 16.0 % |
| 2 | 1.539 mT | 1.532 mT | 1.217 mT | 1.953 mT | 16.4 % |
| 3 | 1.485 mT | 1.482 mT | 1.152 mT | 1.912 mT | 17.5 % |
| 4 | 1.348 mT | 1.342 mT | 1.018 mT | 1.776 mT | 18.9 % |
| 5 | 1.160 mT | 1.153 mT | 0.853 mT | 1.577 mT | 20.5 % |

An error-field amplitude from these coils therefore carries 16 %, and one inference
accounts for nearly all of it: the n = 1 amplitude correlates with the mounting radius at
−0.96 and with the width at +0.25, and with the other five at under 0.05. Pinning how far
out the coils sit would collapse the bound; the corner rounding and the placement within
a module would not move it.

One measurement stands against the field itself. Flux surface mapping imaged a helical
displacement of the magnetic axis of about 10 cm in the high-iota configuration, and a
compass scan found the trim setting that cancels it, so driving the model's circuits at
that setting has to reproduce a displacement of the same size.
The calibration is made at the planar coil current the setting was published
alongside, −9790 A per turn, and gives 6.7 mm.

That is a factor of fifteen. The axis response to an n = 1 field is resonant, going as one
over the transform's distance from unity, so stepping the planar
current moves the on-axis transform at a fixed field:

| Planar current | ι on axis | \|ι − 1\| | n = 1 displacement | Product |
|---|---|---|---|---|
| −11000 A | 1.07091 | 0.07091 | 4.03 mm | 0.2860 mm |
| −9790 A | 1.03921 | 0.03921 | 6.69 mm | 0.2625 mm |
| −9000 A | 1.02159 | 0.02159 | 13.28 mm | 0.2867 mm |
| −8500 A | 1.01209 | 0.01209 | 29.91 mm | 0.3617 mm |

The displacement moves by a factor of 7.42 across that scan and the product by 1.38, so the
response is the resonance and the field behind it is the 0.286 mm the product measures.
Closer in the axis search returns nothing: on the resonant surface the axis is no longer an
isolated fixed point of the return map.

At that constant the measured 10 cm needs a transform 0.00286 from unity, where the
published planar current puts this model at 0.03921. So what the comparison bounds is which
configuration the mapping was done in, and flux surface mapping of an n = 1 error field is
done where the 1/1 resonance lies inside the plasma.

### The machine's own error field

The trim coils exist to cancel W7-X's intrinsic field error, and the settings that do it
were measured by compass scans over trim amplitude and phase, read out on divertor
thermocouples and on the imaged magnetic topology. `programmes.py` carries them:

| Configuration | Planar current | Trim amplitude | Phase | Read out on |
|---|---|---|---|---|
| Standard | −700 A | 98 A | 162° | divertor thermocouples |
| Narrow mirror | −750 A | 99 A | 163° | divertor thermocouples |
| High iota | −9790 A | 134 A | 180° | divertor thermocouples |
| High iota | −9790 A | 123 A | 174° | axis displacement, flux surface mapping |

The five coils are driven as I_k = A₀ cos(2π(k−1)/5 − φ), phase measured from the module 1
trim coil, which is the convention the settings are published in.

A later campaign minimised the divertor load spread directly, in both field senses and for
two harmonics, the 2/2 one reached with the in-vessel control coils rather than the trim
coils. `programmes.SYMMETRISATION_SETTINGS` carries those:

| Field | Harmonic | Circuit | Current | Spread before | after | Programme |
|---|---|---|---|---|---|---|
| Forward | 1/1 | trim | −109 A | 0.75 | 0.27 | 20240918.036 |
| Forward | 2/2 | control | 480 A | 0.27 | 0.067 | 20240918.051 |
| Reversed | 1/1 | trim | −109 A | 0.48 | 0.45 | 20250218.056 |
| Reversed | 2/2 | control | 480 A | 0.45 | 0.18 | 20250218.064 |

The spread is the relative standard deviation of the load across the divertor, and the
intrinsic amplitude of both harmonics is given as 0.5 × 10⁻⁴. Reversing the field leaves
nearly three times the residual at the same amplitudes. These settings are quoted at phases
of the harmonic itself rather than of the coil waveform, which is a different convention
from the table above.

The model's coils are the as-designed set and carry no intrinsic error, so
`python -m w7x_twin errorfield` drives the negative of a measured correction into it, over a
response table tabulated on the whole torus because an n = 1 waveform is not periodic in
the field period. The divertor load is
then attributed per module from a fan launched at every module and above and below the
midplane.

| Multiple of the measured correction | Module max/min | Module spread |
|---|---|---|
| −2 | 1.341 | 0.113 |
| −1 | 1.192 | 0.068 |
| −0.5 | 1.085 | 0.031 |
| 0 | 1.000 | 0.000 |
| +0.5 | 1.078 | 0.027 |
| +1 | 1.207 | 0.061 |

The fan is anchored to the vacuum boundary the traced field owns and lands eleven
hundred of its twelve hundred lines. The unpowered set spreads the five modules evenly
to the digits the tracer carries, so the table is field and not discretisation. At the
measured amplitude and phase the imbalance between the most and least loaded module is
1.19 to 1, against a measured factor of almost four in temperature rise falling below
two once the correction is applied. The trim filaments are reconstructed, so this
amplitude carries the 16 % of the table above, and the source states that the divertor
misalignment contribution to the measured imbalance could not be separated from the
field's.

Upper against lower is a different asymmetry: with no trim current the most loaded unit
of an element carries 1.5 times the least, which is the launch geometry of a fan rather
than the field.

### The coil deviation it implies

The same field can be carried as a deviation of the
superconducting set rather than as the trim current that cancels it.
The measured correction puts 0.2330 mT into n = 1 on the R = 6.2 m midplane circle, which is
the figure `error_field.py` reports for the same field, and three deviations of the coil set
are driven with an n = 1 pattern around the torus against it:

| Deviation | Response | Equivalent to the measured field |
|---|---|---|
| Each coil displaced along the major radius | 2.01 μT/mm | 116 mm |
| Each coil tilted about the major radius | 1.35 μT/mm | 172 mm |
| Each winding deformed out of its own plane | 1.72 μT/mm | 135 mm |

All three are linear in their amplitude to 3 % over an order of magnitude and agree with each
other to 49 %, so this measure does not distinguish which deviation the machine carries. The
amplitudes it needs sit far above the assembly tolerance: W7-X was built to about a
millimetre, at which a rigid shift on every coil gives 2 μT against the 233 μT the measured
correction carries.

The second harmonic distinguishes them. `python -m w7x_twin intrinsic` scales
five deviations of one module to reproduce the measured 1/1 resonant amplitude, which every
one does by construction, and holds each against the 2/2 it then predicts: the predictions span a factor of 152. The closest is a radial shift of 4.04 mm
predicting half the published 2/2, so the intrinsic field is not a single-module rigid
displacement of any of the five kinds, and the record is
`results/discharges/intrinsic_error_field.json`.

## Neoclassical transport

`neoclassical.py` computes the neoclassical transport from the equilibrium.

The effective ripple of Nemov et al. measures the net radial magnetic drift of trapped
particles integrated over all trapped orbits and sets the 1/ν transport, which scales
as its 3/2 power. It is evaluated with the bounce-averaging implementation in DESC,
whose results are published against the NEO code:

| ρ | s | ε_eff |
|---|---|---|
| 0.15 | 0.02 | 0.641 % |
| 0.33 | 0.11 | 0.623 % |
| 0.51 | 0.26 | 0.646 % |
| 0.68 | 0.47 | 0.803 % |
| 0.86 | 0.74 | 0.885 % |
| 0.95 | 0.90 | 1.408 % |

Between 0.6 % and 1.4 %, rising toward the edge. A classical stellarator of this size
runs an order of magnitude higher.

The monoenergetic transport coefficients come from solving a radially local
drift-kinetic equation with MONKES. D₁₁ drives the heat diffusivity used in the
transport split below, and D₃₁ drives the bootstrap current. Both are convolved over a
Maxwellian, and both feed the equilibrium.

Solved on s = 0.2 at ν/v = 10⁻⁵ m⁻¹ with no radial electric field, against the
reference for W7-X EIM distributed with MONKES:

| Quantity | This equilibrium | Reference, solved here | Published for the reference |
|---|---|---|---|
| ι | 0.869 | 0.862 | 0.862 |
| B₀₀ | 2.390 T | 2.431 T | 2.431 T |
| D₃₁, the bootstrap coefficient | 0.3475 | 0.3578 | 0.3549 |
| D₁₁, radial transport | 0.0605 | 0.0809 | 0.0813 |

The pipeline reproduces the published coefficients to 0.5 % in D₁₁ and 0.8 % in D₃₁, which
places the 25 % difference in the first column with the configuration: this equilibrium is
at ⟨β⟩ = 1.05 % with a different transform and field, and D₁₁ in the 1/ν regime goes as
ε_eff^(3/2), so a 16 % difference in effective ripple accounts for it.

Refining the discretisation costs the dense block-tridiagonal solve, whose blocks are
(N_θ N_ζ)² and whose count is N_ξ:

| N_θ × N_ζ × N_ξ | D₁₁ | D₃₁ | Solve time |
|---|---|---|---|
| 21 × 43 × 100 | 0.06406 | 0.3464 | 4 s |
| 31 × 55 × 140 | 0.06148 | 0.3436 | 32 s |
| 41 × 75 × 180 | 0.06050 | 0.3475 | 196 s |
| 51 × 95 × 220 | 0.06005 | 0.3504 | 914 s |

MONKES carries no OpenMP directives and links the reference BLAS, so its whole cost is
one thread of dense linear algebra. Preloading a threaded MKL over it reproduces the
coefficients to eleven digits at 31 times the speed.

### Radial electric field

The field enters the drift-kinetic equation as E_r/v, so its effect depends on speed
and cannot be reduced to a single value over the energy convolution. The coefficients
are solved on a product grid in collisionality and field and interpolated in both axes.
D₁₁ on s = 0.2:

| ν/v [m⁻¹] | Ê_r = 0 | 10⁻⁴ | 3 × 10⁻⁴ | 10⁻³ |
|---|---|---|---|---|
| 10⁻⁵ | 0.0641 | 0.0391 | 0.0157 | 0.0044 |
| 10⁻⁴ | 0.00792 | 0.00775 | 0.00712 | 0.00494 |
| 10⁻³ | 0.00251 | 0.00251 | 0.00250 | 0.00247 |
| 10⁻² | 0.00359 | 0.00359 | 0.00359 | 0.00360 |
| 10⁻¹ | 0.00819 | 0.00819 | 0.00818 | 0.00818 |

The field suppresses the coefficient by a factor of fifteen at the lowest
collisionality and leaves it untouched at the highest: what it closes off is the 1/ν
regime, by giving trapped particles a poloidal precession that carries them off the
drift orbits the regime is built on.

D₃₁ peaks near ν/v = 3 × 10⁻⁵,
passes through zero near 10⁻³ and is negative above it, so it is interpolated linearly
in value against log ν rather than in the logarithm of its magnitude, which would carry
the sign change across without the zero it passes through. D₁₁ is positive over three
decades and is interpolated in log-log, continued outside the table as the power law
its own end points measure. On the eleven-collisionality table that exponent is −0.9987,
the 1/ν regime the table sits in, measured to a tenth of a percent of exact.

The heat diffusivity is insensitive to how that exponent is measured. Fitting it over the
lowest two, three, four or five entries gives −0.999, −0.994, −0.982 and −0.952, and χ
moves by nothing on axis, 0.009 % at mid-radius and 0.127 % at the edge, because the table
reaches below what the convolution samples: the continuation supplies none of χ at any of
the three, and only 1.6 % at the edge comes from above the table's highest collisionality.

`extrapolated_weight` reports that share. On a 5 × 4 grid reaching ν/v = 10⁻⁵ and
Ê_r = 10⁻³, a 300 eV edge point at
20 kV/m draws 24 % of its χ from beyond the tabulated field and a 2.7 keV core point
0.6 % from below the tabulated collisionality. On the 11 × 6 grid reaching 10⁻⁶ and
3 × 10⁻³ those become 1.9 % and zero, and the core is solved directly.

The table extends along both axes: thirtyfold in the
radial electric field, which takes the field-axis draw of every convolution to zero, and
past nu/v = 3 per metre on the outer surfaces, which leaves the ions drawing at most 2 per
cent from the continuation anywhere. The edge electrons still reach beyond the table, but
the table now ends inside the Pfirsch-Schlueter regime, whose coefficient is linear in
collisionality exactly, and the fitted end exponents on the four outer surfaces are +0.97
to +0.98: what continues past the end is the regime's own law, measured rather than
assumed.

### From monoenergetic coefficients to a heat diffusivity

MONKES strips the drift prefactor m v²/e from the radial magnetic drift and, per
Appendix E of its paper, normalises its output by K₁₁ = (dψ/dr)⁻², which its source
applies as a division by `psi_p**2`. Restoring the two factors of that prefactor and
dividing by the speed the kinetic equation was normalised with gives the physical
monoenergetic coefficient

    D₁₁ [m²/s] = (m² v³ / e²) · D₁₁^MONKES

which is dimensionally consistent, the output carrying units of 1/(T² m). Convolving
over a Maxwellian with the Onsager weight for the temperature gradient,

    χ = (2/√π) ∫ dK √K e^(−K) (K − 3/2)² D₁₁(K),

with the pitch-angle deflection frequency evaluated for each species, gives an
absolute neoclassical heat diffusivity. The coefficient is interpolated in
collisionality from the drift-kinetic table and continued below its lowest entry as
1/ν, which is exact in the regime that table already sits in, and carried to other
flux surfaces by the effective ripple to the 3/2 power.

### The radial dependence of the coefficients

A single surface carried across the profile requires a radial scaling, and the usual one
is the effective ripple to the 3/2 power, the radial dependence of the 1/ν coefficient.
The scan is repeated on twelve surfaces, so the solved dependence
stands against that scaling. Both columns are ratios to the surface nearest s = 0.2:

| s | ε_eff | D₁₁ ν/v | Solved | ε_eff^(3/2) |
|---|---|---|---|---|
| 0.02 | 0.641 % | 6.400 × 10⁻⁷ | 1.02 | 1.04 |
| 0.05 | 0.637 % | 6.361 × 10⁻⁷ | 1.01 | 1.03 |
| 0.10 | 0.626 % | 6.650 × 10⁻⁷ | 1.06 | 1.01 |
| 0.16 | 0.623 % | 6.292 × 10⁻⁷ | 1.00 | 1.00 |
| 0.25 | 0.647 % | 5.828 × 10⁻⁷ | 0.93 | 1.06 |
| 0.35 | 0.707 % | 5.482 × 10⁻⁷ | 0.87 | 1.21 |
| 0.45 | 0.790 % | 5.480 × 10⁻⁷ | 0.87 | 1.43 |
| 0.55 | 0.865 % | 5.912 × 10⁻⁷ | 0.94 | 1.64 |
| 0.65 | 0.905 % | 6.876 × 10⁻⁷ | 1.09 | 1.75 |
| 0.75 | 0.905 % | 8.404 × 10⁻⁷ | 1.34 | 1.75 |
| 0.85 | 1.210 % | 1.019 × 10⁻⁶ | 1.62 | 2.70 |
| 0.95 | 1.408 % | 1.209 × 10⁻⁶ | 1.92 | 3.40 |

The two agree over the inner third of the profile and part company outside it. The solved
coefficient falls to a minimum near s = 0.4 and rises outward from there, while the
effective ripple rises monotonically: at the edge the scaling overstates the coefficient
by 77 %, at mid-radius by 40 to 74 %. Everything below uses the solved profile.

The Boozer file covers every interior surface: covering only s ≥ 0.04 leaves MONKES
extrapolating below it, which puts the coefficient 40 % high at s = 0.02, 12 % high at
s = 0.05, and manufactures a rise toward the axis.

That profile is solved at 31 × 55 × 140. Three of its surfaces repeated at 41 × 75 × 180
separate the resolution from the radial dependence, a resolution error common to every
surface cancelling in the ratios:

| s | D₁₁ at 31 × 55 × 140 | at 41 × 75 × 180 | Change | Ratio to s = 0.16, coarse | fine |
|---|---|---|---|---|---|
| 0.05 | 7.2623 × 10⁻² | 7.2210 × 10⁻² | −0.6 % | 1.142 | 1.151 |
| 0.16 | 6.3616 × 10⁻² | 6.2720 × 10⁻² | −1.4 % | 1.000 | 1.000 |
| 0.85 | 1.0279 × 10⁻¹ | 1.0225 × 10⁻¹ | −0.5 % | 1.616 | 1.630 |

The absolute coefficients move by under 1.5 % and the ratios by under 0.9 %: the edge ratio
is 1.62 coarse and 1.63 fine against the scaling's 2.70. D₃₁ is the more
resolution-sensitive of the two, moving 1.1 % at s = 0.16 and 5.1 % at s = 0.85.

### Across configurations and pressures

That disagreement could belong to the configuration, to the pressure or to the scaling. Four cases are prepared, each with
its own Boozer file and its own effective-ripple profile, and MONKES
solves the coefficient on twelve surfaces of each.

Within a case, does the coefficient's ratio between surfaces follow the ratio of
ε_eff^(3/2)? That is what carrying one surface's table across the profile assumes.

| Case | ⟨β⟩ | ε_eff on axis | at edge | Scaling over solved, median | range |
|---|---|---|---|---|---|
| Standard, vacuum | 0 % | 0.835 % | 1.121 % | 1.34 | 0.97 to 1.68 |
| Standard | 1.05 % | 0.641 % | 1.408 % | 1.35 | 0.95 to 1.77 |
| Standard | 2.08 % | 0.454 % | 1.324 % | 1.32 | 0.93 to 1.65 |
| High mirror | 1.11 % | 2.629 % | 3.529 % | 1.10 | 0.98 to 1.24 |

The scaling overstates the radial dependence in every case. Pressure does not account for
it: the median factor is 1.34, 1.35 and 1.32 across a vacuum field, 1 % and 2 % β, while
the effective ripple on axis halves over that range. The configuration accounts for part of
it: high mirror carries four times the ripple and a far flatter ripple profile, and there
the scaling is 10 % out rather than 35 %.

Between cases at one surface, does the coefficient's ratio follow the ripple ratio? That is
what carrying one configuration's table to another assumes.

| Case | ε_eff at s = 0.16 | D₁₁ ν/v | Measured ratio | ε_eff^(3/2) | Scaling over measured |
|---|---|---|---|---|---|
| Standard, vacuum | 0.769 % | 8.510 × 10⁻⁷ | 1.35 | 1.37 | 1.01 |
| Standard, ⟨β⟩ = 1.05 % | 0.623 % | 6.292 × 10⁻⁷ | 1.00 | 1.00 | 1.00 |
| Standard, ⟨β⟩ = 2.08 % | 0.504 % | 4.640 × 10⁻⁷ | 0.74 | 0.73 | 0.99 |
| High mirror, ⟨β⟩ = 1.11 % | 2.631 % | 5.565 × 10⁻⁶ | 8.84 | 8.68 | 0.98 |

Here it is exact to 2 % over a factor of nine in the coefficient, high mirror included.

The scaling holds for the amplitude of the ripple-driven drift at one place and fails for
its radial variation within one configuration. Between configurations at a fixed surface the remaining
geometric factors are alike and the ripple carries the difference; across surfaces they
diverge and it does not.

`transport.py` holds the neoclassical diffusivity fixed and scales only the remainder to
reproduce ISS04, iterating because that diffusivity depends on the temperature the
balance is solving for. At 5 MW with no radial electric field:

| s | T_e | χ_neo | χ_anom | Neoclassical share | From outside the table |
|---|---|---|---|---|---|
| 0.01 | 2681 eV | 0.645 m²/s | 0.251 m²/s | 72.0 % | 0.0 % |
| 0.13 | 2228 eV | 0.264 m²/s | 0.367 m²/s | 41.9 % | 0.0 % |
| 0.25 | 1684 eV | 0.134 m²/s | 0.483 m²/s | 21.7 % | 0.1 % |
| 0.49 | 902 eV | 0.043 m²/s | 0.715 m²/s | 5.7 % | 0.9 % |
| 0.97 | 146 eV | 0.005 m²/s | 1.179 m²/s | 0.5 % | 5.9 % |

Neoclassical transport dominates the hot core and collapses outward; the edge is
entirely anomalous. Across heating power the on-axis share rises
61.5 % → 72.0 % → 79.0 % → 84.7 % for 2, 5, 10 and 20 MW, the 1/ν regime's defining
behaviour: χ_neo goes as T^(7/2) while the anomalous remainder grows far more slowly.

### Confinement anchor

The rows above sit at ISS04 exactly, and W7-X is reported above the scaling. The same
solve at 1.3 × ISS04:

| ISS04 × | P | W | T_e(0) | χ_neo(0) | χ_anom(0) | Neoclassical share |
|---|---|---|---|---|---|---|
| 1.0 | 5 MW | 0.846 MJ | 2681 eV | 0.645 m²/s | 0.251 m²/s | 72.0 % |
| 1.3 | 5 MW | 1.099 MJ | 3114 eV | 1.055 m²/s | 0.167 m²/s | 86.3 % |

At the enhanced confinement the on-axis share runs 79.7 % → 86.3 % → 90.2 % → 93.2 %
across 2, 5, 10 and 20 MW. Taking the scaling at unity therefore depresses the
temperature, inflates the anomalous remainder and biases the neoclassical share
downward by fourteen points at 5 MW.

### Effect of an imposed radial electric field

At 5 MW, holding everything else fixed:

| E_r | T_e(0) | χ_neo(0) | χ_anom(0) | Share at ISS04 × 1.0 | at × 1.3 |
|---|---|---|---|---|---|
| 0 | 2681 eV | 0.645 m²/s | 0.251 m²/s | 72.0 % | 86.3 % |
| 5 kV/m | 2847 eV | 0.305 m²/s | 0.258 m²/s | 54.1 % | 71.1 % |
| 10 kV/m | 2860 eV | 0.267 m²/s | 0.259 m²/s | 50.8 % | 66.8 % |
| 20 kV/m | 2880 eV | 0.234 m²/s | 0.259 m²/s | 47.4 % | 62.8 % |

The field more than halves the core neoclassical diffusivity and takes the neoclassical
share from 72 % to 47 %, most of it by 5 kV/m. The anomalous remainder barely moves: the
field acts on the computed channel alone. At the enhanced anchor the same field leaves
the neoclassical channel still dominant at 63 %.

### Ambipolar radial electric field

The rows above impose a uniform field. Ambipolarity fixes it instead: at each surface the
electron and ion radial particle fluxes are computed from the same coefficients the heat
channel uses, and the field is where they balance. `python -m w7x_twin efield` solves two
operating points on the temperature profile the power balance returns at 5 MW and
1.3 × ISS04: the package's high-performance profile, and a low-density point at
2.0 × 10¹⁹ m⁻³ on axis.

| s | T_e | T_i | a/L_n | a/L_Te | Roots | E_r |
|---|---|---|---|---|---|---|
| 0.05 | 4194 eV | 2307 eV | 0.00 | 1.24 | −2.55 kV/m | −2.55 kV/m |
| 0.15 | 3105 eV | 1708 eV | 0.00 | 2.38 | −6.53 kV/m | −6.37 kV/m |
| 0.25 | 2287 eV | 1258 eV | 0.01 | 3.02 | −7.41 kV/m | −7.41 kV/m |
| 0.41 | 1433 eV | 788 eV | 0.09 | 3.62 | −5.67 kV/m | −5.69 kV/m |
| 0.55 | 973 eV | 535 eV | 0.34 | 4.07 | −4.26 kV/m | −4.26 kV/m |
| 0.69 | 656 eV | 361 eV | 1.05 | 4.92 | −3.61 kV/m | −3.61 kV/m |
| 0.85 | 377 eV | 208 eV | 4.01 | 8.04 | −4.13 kV/m | −4.13 kV/m |

One root at every surface, all negative: the ion root, which is where a stellarator at
this density sits. The field peaks at −7.4 kV/m near s = 0.25 and weakens toward the
axis. The imposed rows apply their largest suppression where the solved field is
smallest.

At 2.0 × 10¹⁹ m⁻³ the same balance drives the core electron temperature to 8.4 keV and
the root structure changes:

| s | T_e | T_i | Roots | E_r |
|---|---|---|---|---|
| 0.05 | 8361 eV | 4599 eV | −0.87, +0.88, +9.01 kV/m | +9.01 kV/m |
| 0.15 | 6139 eV | 3376 eV | −1.08, +1.25, +12.44 kV/m | +12.16 kV/m |
| 0.25 | 4474 eV | 2461 eV | −2.94 kV/m | −2.94 kV/m |
| 0.41 | 2738 eV | 1506 eV | −7.58 kV/m | −7.28 kV/m |
| 0.55 | 1805 eV | 993 eV | −7.83 kV/m | −7.83 kV/m |
| 0.69 | 1167 eV | 642 eV | −7.04 kV/m | −7.19 kV/m |
| 0.85 | 613 eV | 337 eV | −7.98 kV/m | −7.98 kV/m |

The centre carries the electron root at +9 to +12 kV/m behind an unstable middle root
and falls back to the ion root from s = 0.25 outward, which is the core field structure
of the machine's low-density discharges, positive at this order where they are measured.
Every surface row in the record carries all of its roots and the branch the chosen field
sits on.

Carried into the power balance, the solved field gives an on-axis neoclassical share of
63.6 % at ISS04 and 80.3 % at 1.3 × ISS04, between the no-field and uniform-5 kV/m cases
and nearer the former.

Boozer coordinates are produced with `booz_xform`. Its NetCDF writer leaves `phi_b`, the
toroidal flux profile, filled with zeros, and MONKES returns NaN for every flux-surface
quantity without it, so `equilibrium.py` copies the profile in from the VMEC output.

Both codes are external. DESC is installed in its own environment; MONKES is built
from source and is distributed by CIEMAT free of charge to individuals and institutions
not operating for profit, with commercial use requiring prior permission from its
authors. Neither is vendored here.

## Bootstrap current

`current.py` evaluates the Redl formula [4] from the kinetic profiles and the
equilibrium geometry, then iterates the knots of the VMEC spline current profile by
Newton's method until the equilibrium reproduces the ⟨**J**·**B**⟩ its own profiles
imply. The finite-difference Jacobian of ⟨**J**·**B**⟩ with respect to the knot values
is rebuilt at each outer step.

The Redl formula is parametrised by the toroidal mode number of the field's symmetry
direction, which is read off the equilibrium's |B| spectrum: for W7-X the helical
(m = 1, n = N_fp) component is −0.057 B₀₀ against
−0.023 B₀₀ for the axisymmetric (m = 1, n = 0) component, so the helical branch
applies. Evaluated on the axisymmetric branch the same profiles return 209 kA, an
order of magnitude above the machine's operating range.

At n_e(0) = 8 × 10¹⁹ m⁻³, T_e(0) = 3.5 keV, T_i(0) = 1.8 keV, corresponding to
⟨β⟩ = 1.05 %, the converged current is 12.86 kA. This lies within the 0–43 kA range
spanned by the machine's own OP1.2a current-mimic configurations.

The direction follows from the machine's own current-mimic tapers, which reproduce, with
the planar coils alone, the equilibrium a given bootstrap current would produce. Their
edge transform rises with the current they stand in for:

| Mimicked current | ι on axis | ι at edge |
|---|---|---|
| 0 kA | 0.78395 | 0.87836 |
| 11 kA | 0.79524 | 0.89470 |
| 22 kA | 0.80747 | 0.91259 |
| 32 kA | 0.82045 | 0.93143 |
| 43 kA | 0.83393 | 0.95157 |

A bootstrap current must therefore raise the edge transform here too, and both drives
do: at ⟨β⟩ = 1.05 % the pressure alone gives ι_edge = 0.95288, and the self-consistent
current carries it to 0.98084. Pressure by itself accounts for 0.0008 of that; the
current accounts for 0.028, and reversing the current reverses the shift exactly.
`python -m w7x_twin bootstrap` exits non-zero if the direction ever disagrees.

The sign of the toroidal flux divides the thermodynamic drives inside the formula, so the
wrong sign reverses the current while leaving its magnitude, its convergence and its
residual unchanged.

The formula was derived for quasisymmetric fields and W7-X is quasi-isodynamic, so it
remains an analytic estimate whichever symmetry direction it is evaluated along.

### From the drift-kinetic coefficient

Passing `target="drift_kinetic"` replaces the formula with the D₃₁ coefficient of the
monoenergetic solution. Undoing MONKES's normalisation by (dψ/dr) and by B₀₀ and the
stripped drift prefactor, the parallel flow moment gives

    ⟨J·B⟩_a = p_a B₀₀ (2/√π) ∫ dK √K e^(−K) K D₃₁(K) [A₁ + (K − 3/2) A₂]

with A₁ the density gradient less the electric-field term and A₂ the temperature
gradient, both logarithmic and per metre; the factor (dψ/dr) cancels between the
normalisation and the change of drive from ψ to r.

MONKES reconstructs a right-handed frame from the Boozer file by reversing the toroidal
direction, and reports ι = −0.869 where VMEC reports +0.869. The parallel flow moment
carries no compensating factor: the two drives agree in direction as they stand.

| Drive | Converged current | Mismatch | ι at edge |
|---|---|---|---|
| Redl analytic formula | −12.86 kA | 4.4 % | 0.98084 |
| Drift-kinetic D₃₁, twelve surfaces | −10.16 kA | 1.8 % | 0.97500 |

Both sit inside the 0–43 kA range spanned by the machine's own current-mimic
configurations, both raise the edge transform, and the two agree to 21 % in magnitude at
this scenario.

That agreement does not carry. Both routes are run across
two configurations and five scenarios, the temperature scale setting β and the density
scale moving the collisionality at fixed profile shape:

| Configuration | Scenario | ⟨β⟩ | Redl | D₃₁ | Ratio |
|---|---|---|---|---|---|
| Standard | half power | 0.55 % | −6.85 kA | +0.67 kA | −0.10 |
| Standard | reference | 1.05 % | −12.86 kA | −10.17 kA | 0.79 |
| Standard | hot | 1.45 % | −17.34 kA | −26.18 kA | 1.51 |
| Standard | dense | 1.59 % | −19.36 kA | −8.76 kA | 0.45 |
| Standard | dilute | 0.63 % | −7.64 kA | −9.95 kA | 1.30 |
| High mirror | half power | 0.58 % | −8.13 kA | +0.67 kA | −0.08 |
| High mirror | reference | 1.11 % | −15.40 kA | −10.30 kA | 0.67 |
| High mirror | hot | 1.53 % | −20.99 kA | −26.53 kA | 1.26 |
| High mirror | dense | 1.68 % | −23.29 kA | −8.88 kA | 0.38 |
| High mirror | dilute | 0.66 % | −9.12 kA | −10.08 kA | 1.11 |

The median ratio is 0.73 and the range runs from −0.10 to 1.51, so the two routes disagree
by 27 % in the median and by 110 % at worst. They cross near the reference scenario and
diverge in both directions away from it: the drift-kinetic drive rises faster with
temperature and falls faster with density than the analytic one. At half power it nearly
vanishes and changes sign, which follows from D₃₁ itself changing sign with collisionality,
so a colder plasma moves the convolution into the negative branch.

Regressing the ratio against the two variables the scenarios move separates them. Beta
leaves 97 % of its spread unexplained at a slope of 31.6; the logarithm of the
collisionality leaves 21 % at a slope of −1.61. Beta mixes density and temperature and D₃₁
changes sign with collisionality, so what separates the two routes is where the Maxwellian
convolution sits relative to that sign change.

How D₃₁ is carried across the profile barely matters: solved on twelve surfaces it gives
−10.16 kA, solved on one and held flat, −10.13 kA. The effective-ripple scaling does
matter. That exponent belongs to the 1/ν coefficient, and applying it to the bootstrap
coefficient amplifies the cold edge, where D₃₁ has already changed sign, by a factor
above three, so the edge cancels most of the core and the converged current falls to
6.8 kA.

The disagreement is decomposed. Each input
the analytic formula carries and the monoenergetic route did not is switched on in turn,
on one equilibrium: the profile's effective charge closes none of the gap, the ambipolar
radial field closes 6 %, restoring electron momentum widens it, and all three together
leave the median gap where it started. What remains is where the coefficient is read:
with the field axis of the table extended thirtyfold the drift-kinetic current collapses
further, because a resolved ambipolar field suppresses the 1/ν channel more strongly.
Both routes are then diffused through the plasma's own resistivity, whose fill-in time of
half a second makes the six-second measurement nearly saturated, and held against the one
measured toroidal current: the Redl route reaches −8.9 kA and the drift-kinetic one under
1 kA against the measured 5.0. Record: `results/plasma/bootstrap_routes.json`.

Reproduce with `./run.sh -m w7x_twin bootstrap`, which runs both drives against
the same equilibrium and profiles and checks the direction against the machine.

## Transport, bootstrap and equilibrium converged together

Solved in sequence, the power balance runs on a fixed equilibrium, the bootstrap current
on prescribed profiles, and the equilibrium on a pressure neither produced. Each of the
three depends on the other two, so `current.py` iterates them, profiles to bootstrap current and
pressure to equilibrium to profiles, until the stored energy and the net current stop
moving.

| Heating | W | sequential | I_boot | sequential | ⟨β⟩ | sequential | ι_edge | sequential |
|---|---|---|---|---|---|---|---|---|
| 2 MW | 0.780 MJ | 0.769 MJ | −9.81 kA | −12.86 kA | 0.86 % | 1.05 % | 0.97462 | 0.98084 |
| 5 MW | 1.117 MJ | 1.099 MJ | −13.47 kA | −12.86 kA | 1.23 % | 1.05 % | 0.98166 | 0.98084 |
| 10 MW | 1.465 MJ | 1.441 MJ | −16.94 kA | −12.86 kA | 1.62 % | 1.05 % | 0.98820 | 0.98084 |

Three outer iterations close it at each power, to 2 × 10⁻⁵ in stored energy and
4 × 10⁻³ in current. The sequential columns are the first iteration, which reproduces the
sequential solve exactly.

Stored energy moves by 1.4 to 1.7 %, the confinement scaling setting it either way. The
current and the pressure move. Solved in sequence the bootstrap current is
12.86 kA at every heating power, running on profiles that carry no knowledge of the
power; coupled it follows the power, so the sequential value overstates it by 31 % at
2 MW and understates it by 24 % at 10 MW. ⟨β⟩ moves with it, 0.86 % to 1.62 % against a
fixed 1.05 %.

That reaches the machine through the edge transform. Coupled it runs 0.97462 to 0.98820
across the power scan, so the island chain the divertor is built around moves with the
heating power; solved in sequence it sits at 0.98084 whatever the power.

The bootstrap mismatch rises from 4.4 % on the prescribed profiles to 6.5 % on the solved
ones. The solved temperature profile is steeper at the edge, and six knots of the VMEC
spline current profile resolve the prescribed shape more closely than the solved one.

Reproduce with `./run.sh -m w7x_twin coupled 2 5 10`.

## Machine quantities as intervals

A converged free-boundary solve costs a few seconds here, so the derived quantities can
be reported as distributions. `python -m w7x_twin ensemble` samples
the coil currents within a 0.1 % per-circuit setting tolerance, the kinetic
temperatures within 5 % and the heating power within the 5 % the sources state, solves
every sample, and carries each one down the pipeline: the power balance runs on the
sample's own perturbed profiles at its own perturbed power, and the Redl bootstrap on
its own geometry, so the intervals below are conditioned on everything the equilibrium
ones are. One hundred and twenty-eight samples at ⟨β⟩ = 1.05 %:

| Quantity | Median | Its error | 5th | 95th | Relative spread |
|---|---|---|---|---|---|
| Transform on axis | 0.85514 | 6.4 × 10⁻⁵ | 0.85449 | 0.85571 | 0.047 % |
| Transform at edge | 0.95288 | 2.7 × 10⁻⁵ | 0.95234 | 0.95331 | 0.032 % |
| Aspect ratio | 11.231 | 7.3 × 10⁻⁴ | 11.221 | 11.243 | 0.063 % |
| Plasma volume | 26.444 m³ | 4.4 × 10⁻³ | 26.387 | 26.498 | 0.138 % |
| Field on axis | 2.3140 T | 3.9 × 10⁻⁴ | 2.3090 | 2.3188 | 0.141 % |
| Mirror term | 3.613 % | 9.1 × 10⁻³ | 3.504 | 3.720 | 1.816 % |
| Magnetic well | 3.296 % | 2.6 × 10⁻² | 3.060 | 3.527 | 4.189 % |
| Stored energy | 0.954 MJ | 1.0 × 10⁻² | 0.849 | 1.083 | 7.601 % |
| ⟨β⟩ | 1.048 % | 1.1 × 10⁻² | 0.931 | 1.190 | 7.642 % |
| Stored energy, power balance | 1.185 MJ | 3.5 × 10⁻³ | 1.144 | 1.224 | 2.228 % |
| Confinement time | 0.2366 s | 1.1 × 10⁻³ | 0.2221 | 0.2526 | 3.758 % |
| Electron temperature on axis | 4551 eV | 15 | 4333 | 4761 | 2.837 % |
| Bootstrap current, Redl | −13.93 kA | 1.6 × 10⁻² | −15.38 | −12.51 | 5.804 % |

The error column is the bootstrap standard error of the median, which bounds the digits an
interval supports: the edge transform is resolved to five decimals, β to three.

The edge transform, which sets where the island chain sits, is the most robust of them at
0.032 %, an order below the mirror term and two below the magnetic well. Coil setting
error at this level moves the field shaping without moving the divertor. β and stored
energy carry the profile uncertainty rather than the coil one, and the balance's stored
energy spreads a third as far as the equilibrium's: the confinement anchor is stiffer
against profile perturbations than a prescribed pressure, since the scaling reads the
density at half a power and the heating at a third. The saturated bootstrap spreads
5.8 %, which is what the pressure uncertainty is worth to the current the island
position depends on.

The exhaust chain carries intervals the same way: the traced mapping is fixed, so the
stated input uncertainties, the heating power at five per cent, the perpendicular
diffusivity at its published fifty and the effective charge at its twenty, are sampled
through the layer closure and the deskewed deposition, sixty-four samples at a few
tenths of a second each. The strike-line width comes back 34.9 mm spanning 28.7 to
43.1, the whole interval inside the fitted 2 to 4 cm; the wetted area 1.02 m² spanning
0.98 to 1.06 about the machine's 1; and the net peak 6.3 MW/m² spanning 4.9 to 7.1,
the measured 6 sitting mid-interval. The record carries them under `intervals` in
`results/exhaust/heat_flux.json`.

These are finite-β figures, so the mirror term and the magnetic well are not comparable
with the vacuum entries in the Verification table; pressure deepens the well and reduces
the mirror.

## Convergence of the tracer

Everything traced rides on a fixed-step integration, so the step is refined until the
answers stop moving:

| Steps per field period | Axis R | Its error | ι at 0.92 of the boundary | Its error |
|---|---|---|---|---|
| 30 | 5.947440 m | 17.4 µm | 0.938548 | 1.9 × 10⁻⁴ |
| 60 | 5.947409 m | 13.1 µm | 0.938652 | 3.0 × 10⁻⁵ |
| 120 | 5.947425 m | 2.8 µm | 0.938644 | 1.9 × 10⁻⁵ |
| 240 | 5.947422 m | 0.2 µm | 0.938646 | 2.1 × 10⁻⁵ |
| 480 | 5.947422 m | — | 0.938648 | — |

The model runs at 120, where the axis is 2.8 µm and the transform 1.9 × 10⁻⁵ from the
refined answer.

Wall intersection sets the strike position and is tested at every step, not every nth. A
line covers 80 mm along the field per step, so a coarser test carries that far in ignorance
of where it landed:

| Wall test interval | Lines struck | Mean strike R | Against every step |
|---|---|---|---|
| 16 steps | 24 | 5.653 m | +524 mm |
| 8 steps | 24 | 5.478 m | +350 mm |
| 4 steps | 16 | 5.340 m | +212 mm |
| 2 steps | 16 | 5.159 m | +30 mm |
| 1 step | 16 | 5.129 m | — |

The floor is the poloidal projection of one step at the strike, 22 to 25 mm, which is
quoted with every strike displacement below.

## Effect of the finite conductor build

Replacing each winding-pack centre by one filament per conductor turn, 6120 filaments
against 70, changes the vacuum field on the magnetic axis by 1.21 mT, 507 ppm. The
field magnitude and the plasma geometry are insensitive to it, but the rotational
transform is not.

| Quantity | Single filament | Finite build | Relative change |
|---|---|---|---|
| B on axis | 2.360034 T | 2.360272 T | 1.0 × 10⁻⁴ |
| Plasma volume | 26.21792 m³ | 26.21616 m³ | 6.7 × 10⁻⁵ |
| Transform on axis | 0.84959 | 0.86251 | 1.5 × 10⁻² |
| Transform at edge | 0.95370 | 0.96602 | 1.3 × 10⁻² |
| Mirror term | 4.3945 % | 4.3728 % | 5.0 × 10⁻³ |
| Magnetic well | 1.0133 % | 1.0440 % | 3.0 × 10⁻² |

A 1.3 % shift in edge transform displaces the island chain the divertor is built
around, so the single-filament model is adequate for field strength and plasma shape
and inadequate where the position of a low-order rational matters.

## Coil deflection under load

Every filament segment of an energised coil sits in the field of all the others and feels
I dl × B. `coils.py` sums that over the machine: a non-planar coil at 1.40
MA-turns carries 10.3 MN in total and 2.4 MN/m at its worst point.

The casing is the structural member and the span between supports enters the deflection as
a fourth power, so the pattern is taken from the force and
scans the amplitude:

| Peak deflection | Axis R | Transform on axis | at edge | Volume | Mirror |
|---|---|---|---|---|---|
| cold | — | 0.849585 | 0.953704 | — | — |
| 1 mm | −0.034 mm | 0.848588 | 0.952928 | −0.002 % | +0.054 % |
| 3 mm | −0.103 mm | 0.846604 | 0.951371 | −0.006 % | +0.161 % |
| 10 mm | −0.339 mm | 0.839907 | 0.945964 | −0.017 % | +0.501 % |

The edge transform moves 7.8 × 10⁻⁴ per millimetre of peak deflection. A steel casing of
the published pack section over a 1.5 m span gives 0.23 mm, which is 1.8 × 10⁻⁴ in
transform: below what the finite conductor build is worth and comparable to what the coil
current setting tolerance gives.

## Verification

`python -m w7x_twin validate` writes every comparison below to `results/validation.json` with
the geometry version it was produced from, and exits non-zero if any of them stops
agreeing with its reference. All thirty-two agree.

The one published W7-X equilibrium reconstruction with consistent uncertainties, Koeberl
et al. (MaxEnt 2023, Zenodo 8095035), is carried as its own benchmark: the twin solved at
the reconstruction's currents, pressure profile and toroidal flux reproduces its MAP
equilibrium to +0.02 % on the axis position, -0.45 % and +0.27 % on the transform at axis
and edge, and 0.01 % on minor radius, volume and beta.
`python -m w7x_twin koeberl`, record `results/benchmarks/koeberl.json`.

Standard configuration unless stated.

| Quantity | Model | Published |
|---|---|---|
| Field periods | 5 | 5 |
| Superconducting coils | 50 non-planar, 20 planar | 50, 20 |
| Turns per coil | 108, 36 | 108, 36 |
| Major radius | 5.5153 m | 5.5 m |
| Minor radius | 0.4907 m | ≈ 0.5 m |
| Aspect ratio | 11.24 | ≈ 11 |
| Plasma volume | 26.22 m³ | 26–30 m³ |
| \|B\| on axis, bean plane, 12883 A/turn | 2.5013 T | 2.5 T |
| Rotational transform, axis to edge | 0.8496 → 0.9537 | ≈ 0.85 → below 5/5 |
| Mirror term | 4.39 % | ≈ 5 % |
| Mirror term, high-mirror configuration | 10.05 % | ≈ 10 % |
| Magnetic well | 1.01 % | ≈ 1 % |
| Stored energy at ⟨β⟩ = 1.09 % | 0.997 MJ | ≈ 1 MJ |
| Modular current for 2.5 T averaged on axis | 13 771 A/turn, 1.487 MA-turns | ≈ 1.5 MA-turns |

The two field entries use the two conventions under which 2.5 T on axis is quoted: the
bean plane carries the strongest field along the axis, and the toroidal average over a
field period is approximately seven percent lower.

Independent checks that share no machinery with the equilibrium solver: field-line
tracing locates the magnetic axis at R = 5.94742 m carrying 2.5013 T, and the traced
vacuum rotational transform agrees with the VMEC `iotaf` profile to better than 1 %
across the profile and exactly at mid-radius.

Checks on the added circuits and derived quantities:

| Check | Result |
|---|---|
| Superconducting field with added circuits unpowered | unchanged, 0.000 × 10⁰ T |
| Toroidal harmonics n = 1–4 with no trim current | 6 × 10⁻¹⁶ T |
| n = 1 harmonic, `trim_a1` at 1800 A/turn | 2.64 × 10⁻⁴ T, 110 ppm of B₀ |
| n = 5 and n = 10, `trim_a1` at 1800 A/turn | unchanged to five digits |
| Vessel envelope | R 4.298–6.466 m, Z ± 1.283 m |
| Connection length, bean plane | 2090 m at the separatrix, 3.0 m median in the SOL |
| Self-consistent bootstrap at ⟨β⟩ = 1.05 % | 12.8 kA |
| Finite build against single filament, on-axis field | 507 ppm |

## Field of the plasma currents

Tracing the coil field alone gives the island chain the machine has before a plasma
exists. `plasma_response.py` adds the field of the currents flowing in the plasma, so
the chain is computed in the total field at finite β. The contribution is obtained by
integrating Biot-Savart over the plasma volume using the equilibrium's own current
density.

The surface form, the virtual casing integral over **n** × **B**, requires an O(|B|) sheet
field to cancel down to an O(β|B|) result: a converged quadrature of it still left the
boundary normal field an order of magnitude worse than the vacuum field alone. The volume
current is itself of order β and varies smoothly, so it carries no such cancellation.

VMEC holds the current density already multiplied by the Jacobian in a left-handed
coordinate system; with that resolved the reconstructed net toroidal current reproduces
`ctor` exactly, 0.0 A against 0.0 A and 12800.0 A against 12800.0 A. The on-axis field is then diamagnetic
without further adjustment, and scales linearly with β:

| ⟨β⟩ | \|B\| plasma on axis | \|B\| vacuum | \|B\| total | \|B\| from the equilibrium |
|---|---|---|---|---|
| 0.063 % | 0.0052 T | 2.5219 T | 2.5205 T | 2.5269 T |
| 0.551 % | 0.0266 T | 2.5058 T | 2.4793 T | 2.4936 T |
| 1.049 % | 0.0520 T | 2.4898 T | 2.4379 T | 2.4602 T |
| 2.075 % | 0.1045 T | 2.4582 T | 2.3544 T | 2.3934 T |

The integrand is regular outside the plasma and singular within it, so one refinement
does not serve both, and the volume is sampled at three places. Changes in |B| between successive
samplings, at ⟨β⟩ = 1.05 %:

| Elements | On the axis | At the boundary | 0.1 m outside it |
|---|---|---|---|
| 33 600 | 27.744 mT | 5.310 mT | 1.404 mT |
| 140 400 | 32.412 mT | 1.774 mT | 2.004 mT |
| 480 000 | 34.211 mT | 0.802 mT | 1.890 mT |
| 2 160 000 | 35.329 mT | 1.725 mT | 1.853 mT |
| Residual | 1.117 mT, 3.2 % | 0.923 mT | 0.036 mT, 2.0 % |

The island region converges: 2.0 % at the finest sampling, against 2.4 T of total field.
The boundary does not, which is where the integrand is singular and the evaluation point
sits on the source surface. The axis is still climbing monotonically at the finest
sampling, 3.2 % per refinement, so the on-axis field is a lower bound.

Tracing the total field puts the magnetic axis at R = 5.97619 m against 5.94742 m in the
coil field alone, a displacement of +28.76 mm at ⟨β⟩ = 1.05 %. The equilibrium's own
Shafranov shift at the same pressure is +29.12 mm, so the two agree to 1.2 %.

The volume sum runs through `plasma_response.py` on a CUDA device when one is
available, which makes the sampling that resolves the boundary region affordable: 1.92 million current elements onto 527 076 grid points in 327 s.

Both ends are closed. On the axis the refinement sequence is extrapolated to
zero element size, and the extrapolation carries its own correction beside it: the last
refinement moves the axis value by 3.2 % and the limit sits 6.7 % above it. On the boundary
the integrand is singular and no refinement helps, because the field is discontinuous
across the current that produces it; the virtual-casing sheet gives the one-sided exterior
limit, and with the coil field subtracted on the surface, so the answer is not the small
difference of two large contributions, sheet and volume agree to 5.2 % at eight panel
widths of standoff. The sheet carries the equilibrium's net toroidal current to a part in
five thousand. Traced in the total field the island spans 63.5 mm against 57.1 in the
vacuum field, so an island statement no longer rests on a field the plasma is absent from:
`python -m w7x_twin response`, record `results/magnetics/plasma_response.json`.

## Strike-point attribution

Traced field lines terminate on the plasma-facing components, so each strike is attributed
to one of the ten divertor units: the module from the toroidal angle it landed at, the unit
from which side of the midplane. With a trim coil energised or an as-built error field
present the ten do not receive equal load.

`python -m w7x_twin strikes` repeats the same fan of twelve lines at the equivalent point
of each module, which makes the per-unit tally a periodicity test:

| Element | Lines | Median connection length | Longest |
|---|---|---|---|
| Divertor horizontal target | 30 | 5.2 m | 6.3 m |
| Divertor vertical target | 20 | 52.5 m | 161.0 m |

| Element | 1u | 1l | 2u | 2l | 3u | 3l | 4u | 4l | 5u | 5l |
|---|---|---|---|---|---|---|---|---|---|---|
| Divertor horizontal target | 6 | 0 | 6 | 0 | 6 | 0 | 6 | 0 | 6 | 0 |
| Divertor vertical target | 4 | 0 | 4 | 0 | 4 | 0 | 4 | 0 | 4 | 0 |

Fifty of the sixty lines terminate, all of them on a divertor target. Every module receives ten with zero spread, so the
attribution reproduces the periodicity of the field it was given. The fan is launched on
the midplane, where stellarator symmetry pairs each line with another in the same fan, so
the upper units take all of it; a fan launched off the midplane divides between the two.

The unit and section labels follow IPP's component database, which carries 1620
divertor, 3227 baffle and 4914 heat-shield elements tagged by module, by upper or lower
unit, by section and by tile.

## Linear gyrokinetic stability

The anomalous channel of the transport split is a remainder from the confinement scaling.
Linear flux-tube stability gives the growth-rate spectrum each configuration carries and
its response to the temperature gradient.

`python -m w7x_twin gyrokinetic` writes each configuration's equilibrium as a VMEC output
and runs [stella](https://github.com/stellaGK/stella) on a flux tube spanning five field
periods at s = 0.64, one run per binormal wavenumber and temperature gradient, with two
kinetic species and a density gradient of a/L_n = 1. Growth rates in units of v_ti/a:

| Configuration | a/L_T | k_y = 0.4 | 0.7 | 1.0 | 1.4 |
|---|---|---|---|---|---|
| Standard | 1 | 0.0088 | 0.0339 | 0.0576 | 0.0654 |
| Standard | 2 | 0.0332 | 0.0827 | 0.1229 | 0.1509 |
| Standard | 3 | 0.0561 | 0.1301 | 0.1901 | 0.2300 |
| Standard | 4 | 0.0765 | 0.1741 | 0.2504 | 0.3007 |
| High mirror | 1 | 0.0132 | 0.0308 | 0.0341 | 0.0125 |
| High mirror | 2 | 0.0320 | 0.0738 | 0.1013 | 0.1227 |
| High mirror | 3 | 0.0551 | 0.1205 | 0.1714 | 0.2065 |
| High mirror | 4 | 0.0748 | 0.1653 | 0.2333 | 0.2797 |

Both configurations are unstable at every gradient in that table. The high-mirror spectrum
is the flatter of the two and turns over: at a/L_T = 1 it peaks at k_y = 1 and falls away
by k_y = 1.4, where the standard configuration is still rising.

### Threshold and drive branch

Running the same points below those gradients separates the two
branches, by running the same points with one kinetic ion species and a modified Boltzmann
electron response, which removes the trapped-electron drive and leaves the
ion-temperature-gradient one.

| Configuration | a/L_T | k_y = 0.4 | 0.7 | 1.0 | 1.4 |
|---|---|---|---|---|---|
| Standard | 0 | 0.0041 | 0.0139 | −0.0143 | 0.0371 |
| Standard | 0.25 | 0.0014 | 0.0141 | 0.0113 | 0.0316 |
| Standard | 0.50 | −0.0120 | 0.0132 | 0.0313 | 0.0379 |
| Standard | 0.75 | 0.0063 | 0.0196 | 0.0446 | 0.0532 |
| High mirror | 0 | 0.0067 | 0.0148 | 0.0334 | 0.0448 |
| High mirror | 0.50 | −0.0059 | 0.0232 | 0.0353 | 0.0274 |
| High mirror | 0.75 | 0.0115 | 0.0179 | 0.0294 | 0.0117 |

With no temperature gradient at all the plasma is still unstable, at rates of 0.004 to
0.045. Nothing there is temperature-gradient driven: the only drive left is the density
gradient of a/L_n = 1 the scan carries.

The one-species runs place the other branch:

| Configuration | a/L_T | k_y = 0.4 | 0.7 | 1.0 | 1.4 |
|---|---|---|---|---|---|
| Standard | 0.50 | −0.0034 | −0.0019 | −0.0064 | 0.0073 |
| Standard | 1.0 | −0.0940 | 0.0028 | 0.0030 | 0.0028 |
| Standard | 2.0 | 0.0134 | 0.0243 | 0.0505 | 0.0848 |
| Standard | 3.0 | 0.0226 | 0.0636 | 0.1154 | 0.1600 |
| Standard | 4.0 | 0.0377 | 0.1024 | 0.1688 | 0.2181 |
| High mirror | 1.0 | −0.0272 | 0.0091 | 0.0015 | 0.0027 |
| High mirror | 2.0 | 0.0143 | 0.0241 | 0.0491 | 0.0811 |
| High mirror | 4.0 | 0.0378 | 0.1032 | 0.1695 | 0.2127 |

The ion-temperature-gradient branch is stable at a/L_T = 0.5 and marginal at 1, and clearly
unstable at 2, so its threshold lies between 1 and 2 in both configurations. Everything
below that gradient is trapped-electron driven. At a/L_T = 2 and k_y = 1 the two species
give 0.1229 against the ions' 0.0505, so the trapped electrons carry 59 % of the growth
rate at the gradient a machine is operated near.

The ion-only runs also fix the convention: their frequency is positive, so a positive
frequency is the ion diamagnetic direction. Every mode in the first table with a negative
frequency, which is the low-gradient long-wavelength corner, is therefore on the
trapped-electron branch.

Summing γ/k_y² over the spectrum gives the mixing-length estimate of the turbulent
diffusivity, in gyro-Bohm units of 8.40 m²/s at 1.8 keV:

| a/L_T | Standard | High mirror | Difference |
|---|---|---|---|
| 1 | 1.81 | 1.56 | −13.8 % |
| 2 | 4.84 | 4.32 | −10.6 % |
| 3 | 7.76 | 7.28 | −6.1 % |
| 4 | 10.39 | 9.92 | −4.6 % |

High mirror is the quieter configuration throughout, and by most at the low gradients a
machine is operated near. The sum carries a constant that no linear calculation fixes.

### The measured saturation response

Saturation is set by the nonlinear transfer of energy out of the driven modes, so the
same flux tube is run with the nonlinear term included, one run per surface and gradient
across the profile, and `results/turbulence/mixing_length_constant.json` records the
saturated flux of both species against the linear sum at each point. The measurement
refutes the single-constant closure it replaced: the ratio of saturated flux to linear
sum spans a factor of fifty between s = 0.25 and s = 0.81, and the near-threshold runs
saturate at nothing while the linear spectrum still grows. The record is therefore
consumed as a response surface — one curve of flux against linear sum per surface,
anchored at zero by the near-threshold runs — rather than as any constant, with the
electron channel carried as its own curve at 0.30 to 0.42 of the ion flux. A grown
48 × 32 box at the most driven point lowers its flux by a factor above three against the
24 × 16 grid box, so the reader prefers the largest box that ran each point.

A response this steep is a critical-gradient cliff, and a profile-level Picard iteration
falls off it: the two-temperature balance therefore finds each shell's gradient by
bisection on its own flux balance, which settles on the cliff by construction, and the
turbulent channel is suppressed per surface by the Waltz rule where the solved ambipolar
field's shearing rate reaches the fastest linear growth rate.

### Power balance with both channels computed

The spectrum is solved on four surfaces, five gradients and
three wavenumbers, sixty runs, so the mixing-length sum is available wherever the power
balance evaluates it:

| s | a/L_T = 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| 0.25 | 0.121 | 0.422 | 0.717 | 0.984 | 1.867 |
| 0.49 | 0.170 | 0.480 | 0.785 | 1.060 | 1.567 |
| 0.64 | 0.182 | 0.499 | 0.806 | 1.084 | 1.581 |
| 0.81 | 0.190 | 0.524 | 0.833 | 1.115 | 1.624 |

`transport.py` reads that grid at the local gradients and closes it with the measured
response, so `python -m w7x_twin turbulence` runs the two-temperature balance with both
channels computed and nothing anchored to a confinement scaling. The confinement time is
then a result:

| Heating | W | τ_E | ISS04 | τ_E over ISS04 | T_e(0) | χ_neo(0) | χ_turb(0) |
|---|---|---|---|---|---|---|---|
| 2 MW | 0.82 MJ | 0.411 s | 0.298 s | 1.38 | 1796 eV | 0.026 m²/s | 0.002 m²/s |
| 5 MW | 1.32 MJ | 0.265 s | 0.171 s | 1.55 | 3056 eV | 0.083 m²/s | 0.003 m²/s |
| 10 MW | 1.95 MJ | 0.195 s | 0.112 s | 1.74 | 4626 eV | 0.211 m²/s | 0.009 m²/s |

The machine reports its highest-performance operation at 1.4 times the scaling, which the
computed 10 MW row sits just above; the gas-fuelled 0.70 the machine also runs at is what
the balance overshoots, the trending mid-gradient runs still under-measuring the
saturated flux there. Peaking the density by the digitised post-pellet 2.68 moves the
computed confinement by a factor of 1.35 against the 1.86 the machine separates the
regimes by, with the pellet-level density-gradient runs of the queued campaign carrying
the remainder.

### Across flux surfaces and configurations

The same flux tube is run on four surfaces of four
configurations at a/L_T = 2, three wavenumbers each: 48 runs, of which 45 converged.

| Configuration | s = 0.25 | 0.49 | 0.64 | 0.81 |
|---|---|---|---|---|
| Standard | 0.422 | 0.480 | 0.499 | 0.524 |
| High mirror | 0.371 | 0.431 | 0.452 | 0.495 |
| OP1.2a 22 kA | 0.399 | 0.453 | 0.468 | 0.486 |
| OP2 22 kA | — | 0.459 | 0.477 | 0.496 |

Each entry is Σγ/k_y² over k_y of 0.4, 0.7 and 1.0. The drive rises outward in every
configuration, by a quarter from s = 0.25 to 0.81 in the standard case, and the ordering
between configurations holds at every surface: high mirror is the quietest of the four
everywhere, by 12 % at s = 0.25 and 6 % at s = 0.81. The radial variation within one
configuration is larger than the variation between configurations at any one surface.

The three that did not converge are the OP2 22 kA configuration at s = 0.25 at all three
wavenumbers, where stella's VMEC geometry interface fails.

The spectrum is a grid: five flux surfaces, six
temperature gradients reaching below the threshold, four density gradients, three
wavenumbers, on two configurations, 720 of 720 runs returning a growth rate once the
geometry interface reads an equilibrium solved on a doubled radial grid. The density
gradient axis is what lets the turbulent channel tell a peaked profile from a flat one:
peaking the density by the digitised post-pellet 2.68 moves the computed confinement by a
factor of 1.35, against the 1.86 the machine separates the regimes by.
`python -m w7x_twin growth-rate-grid`, record `results/turbulence/growth_rate_grid.json`.

## Density from a particle balance

The temperature profile comes from a power balance. `transport.py` gives the density
the same construction with a particle source in place of the heating and a pinch beside the
diffusion, so the peaking follows from where the particles enter rather than from a chosen
shape.

The pinch is the one free parameter, and the published peaking fixes it:
`python -m w7x_twin density` bisects it against the reported 2.0 for an edge source, which
gives 0.25 m/s inward against a diffusivity of 0.2 m^2/s. What the source position then does:

| Source at s | Peaking | Inside the published 1.2 to 2.8 |
|---|---|---|
| 0.95 | 2.000 | yes |
| 0.80 | 2.169 | yes |
| 0.60 | 3.015 | no |
| 0.40 | 5.708 | no |
| 0.20 | 12.413 | no |
| 0.05 | 19.661 | no |

Gas fuelling is not a free choice of position. Recycling neutrals enter at the separatrix and
are attenuated by the ionisation they undergo, so `neutrals.ionisation_source_profile` places
the source where the density and temperature put it, at a power-weighted s of 0.963. The
profile that source sustains peaks at 1.952, inside the published range, against the 1.055 the
prescribed shape carries.

The pellets are an evolution rather than a choice of position. `transport.evolve_density`
marches the same closure in time: from the flat-topped reference at a peaking of 1.05, a
deposition at s = 0.30 under the calibrated pinch reaches the published 2.0 in 0.19 s,
the timescale the post-pellet phases develop their peaking over, and the record carries
the whole trajectory.

## Carbon and the power it radiates

W7-X puts carbon into its own plasma from graphite limiters and targets, and `kinetics.py`
carries it: an effective charge above one, diluted fuel ions, and heating power that leaves
the plasma as radiation before it reaches a flux surface. The radiative cooling rate is the
Mavrin polynomial fit to coronal-equilibrium ADAS rates, whose coefficients are taken from
the `radas` distribution rather than refitted, and which gives the total electron-impurity
emission, so P = n_e n_C L_Z; the main ions' bremsstrahlung is added from the classical
expression.

The mean charge and the cooling rate are functions of the local electron temperature:
carbon strips fully above about 1 keV, and its cooling rate peaks near 5 eV at
3 × 10⁻³² W m³ and falls two orders by 3 keV. Quasineutrality then fixes the main-ion
density, so a carbon fraction sets the effective charge and the fuel dilution together.
Both fits are piecewise in temperature and their
published coefficients are not constrained to join: the cooling rate steps by up to 7.2 %
at 200 eV and the mean charge by 9.9 % at 3 eV, and `tests/` bounds those steps.

The carbon content is an input, and one discharge set measures it. The 4.7 MW density scan is
reported at a radiated fraction of 0.15 to 0.35, chosen low so the target load follows the
transport rather than the radiation. The model radiates 0.078 of that power at 2 % carbon and
reaches the measured 0.25 at 7.7 %:

| Carbon fraction | Radiated |
|---|---|
| 0 % | 0.000 |
| 1 % | 0.049 |
| 2 % | 0.078 |
| 4 % | 0.137 |
| 8 % | 0.258 |

The 2 % this package runs at therefore radiates a third of what a gas-fuelled discharge did,
and leaves that much more power crossing the separatrix.

## The scrape-off layer and the divertor targets

The tracer says where the power lands and how far it travels. `edge.py` says how
hot it arrives, by parallel conduction from the separatrix, pressure balance along the tube
and the sheath condition at the surface.

The power crossing the separatrix is the
heating less what the impurity radiates. The radial width the power occupies is
`sqrt(chi_perp L / c_s)`, with the diffusivity from the power balance, the connection length
from the tracer and the sound speed from the target solution. The wetted area is the
integral width of the power-weighted arc distribution on each target times the toroidal
length the elements cover, and it depends on that width, so `edge.close_layer`
solves the two against each other under relaxation until they stop moving.

At 5 MW with 2 % carbon: 7.4 % radiated, 4.63 MW crossing the separatrix, the fan
launched at five planes and anchored to the vacuum boundary the traced field owns, a
2.59° incidence, a 24.7 mm decay length, 1.02 m² wetted, and the deskewed deposition
peaking at 6.62 MW/m² over 35.0 mm on the upper vertical target. The divertor concept
quotes local power densities up to 8 MW/m² across its operating range and the actively
cooled divertor is rated at 10 MW/m².

Both the width and the transport behind it are measured. Infrared thermography on a
three-discharge density scan at 4.7 MW fits a strike-line width of 2 to 4 cm across the
divertor, which a boundary code reproduces at a perpendicular particle diffusivity of
0.2 m²/s and three times that in heat. The fan that feeds those rows samples the layer
at five launch planes across one field period, each anchored to its own axis and its
own boundary cut, because the strike line is a band inclined along each target and one
plane sees a single comb of it: sampled fully, the band wets 1.09 m² against the
machine's 1 m², where one plane had left 0.72. The camera rows are then read in the
band's own frame, the drift along each target fitted and removed, since a camera at one
toroidal position sees the local width and not the drifted envelope: the deposition of
`python -m w7x_twin discharge` reads 36.6 mm on the edge value of its own power
balance, 0.92 m²/s, and 34.9 mm at the measured 0.6 m²/s, both inside the fitted band,
and the net peak lands at 6.08 MW/m² against the measured 6.

The deposition resolved along each target's arc separates the mapping from the layer
behind it. At 5 MW the closure's 24.7 mm upstream decay deposits 6.62 MW/m² over 35.0 mm
on the upper vertical target; the same traced fan at imposed decays puts the mapping's
own floor at 16 mm under a 5 mm drift-narrow layer, carries a 10 mm layer to 31 mm, and
saturates every broad layer at 34 to 40 mm, so the deposited width tracks the upstream
layer until the island mapping caps it just above the measured band.

The radiator is resolved along the same arc: every bin of the deposition is its own flux
tube, its own two-point solution and its own Lengyel bound, each bin at its own surface
incidence and at the upstream density its own launch layer supplies, the separatrix
value decayed over the bin's mean launch offset at the particle width. The residence
parameter of the non-coronal cooling rate is formed at the tube's mean Mach-one-half
flow, and each tube is priced at its wetted strip: the strike line crosses a target
diagonally, so a profile binned by arc and averaged toroidally smears one bright moving
strip over many faint bins, and the smear factor — measured per element from the arc
spread within toroidal slices, 3.8 on the horizontal target and 1.0 on the vertical —
restores the parallel flux the strip actually sees with each bin's power conserved.
Priced that way the horizontal band stays attached where the diluted pricing predicted
its collapse, which is the behaviour the infrared measurements show. The carbon the
measured radiated fraction needs is 0.0182 against the charge-implied 0.0169 ± 20 %,
inside the band at +8 %.

Which part of the fan is called the strike line sets that connection length.
The same layer traced in all nine configurations gives a strike-line length whose
median of the struck lines runs 9 to 88 m, the 90th percentile 16 to 185 m and the 99th
16 to 299 m. The published 200 to 300 m therefore corresponds to the far tail rather than
to the bulk of a fan. The model runs at the 90th percentile, where the heat-flux width it
implies is 106 mm against a published 100 mm; taking the 99th instead would raise the width
by a quarter, so the two published quantities do not select the same percentile.

| n_u | T_u | T_t | Regime | Power loss to detach |
|---|---|---|---|---|
| 5 × 10¹⁸ m⁻³ | 82.4 eV | 22.34 eV | conduction limited | 53 % |
| 1 × 10¹⁹ m⁻³ | 82.2 eV | 5.62 eV | conduction limited | 6 % |
| 2 × 10¹⁹ m⁻³ | 82.2 eV | 1.40 eV | detached | none needed |
| 4 × 10¹⁹ m⁻³ | 82.2 eV | 0.35 eV | detached | none needed |

The volumetric loss detachment requires falls as the upstream density rises, which is the
sequence a density ramp walks through. Power and momentum loss enter the model only through
`(1 − f_pow) / (1 − f_mom)`, so losing equal fractions of both leaves the target temperature
unchanged and only thins the target plasma, which is why the scan is over power loss. The recycled flux at the target follows from the solution, and `edge.py` turns it into a
pressure: every arriving ion leaves as an atom at the Franck-Condon energy and is ionised after
a mean free path, so the neutral density is the flux divided by that speed.

| n_u | T_t | Mean free path | Neutral density | Pressure | f_pow | f_mom | T_t with both |
|---|---|---|---|---|---|---|---|
| 5 × 10¹⁸ m⁻³ | 184.7 eV | 70.0 mm | 2.02 × 10¹⁹ m⁻³ | 84 mPa | 0.023 | 0.284 | 229.6 eV |
| 1 × 10¹⁹ m⁻³ | 106.3 eV | 45.2 mm | 3.52 × 10¹⁹ m⁻³ | 146 mPa | 0.040 | 0.329 | 144.1 eV |
| 2 × 10¹⁹ m⁻³ | 41.2 eV | 23.8 mm | 9.06 × 10¹⁹ m⁻³ | 375 mPa | 0.104 | 0.442 | 84.5 eV |
| 4 × 10¹⁹ m⁻³ | 10.6 eV | 18.5 mm | 3.54 × 10²⁰ m⁻³ | 1.46 Pa | 0.405 | 0.754 | 57.3 eV |
| 8 × 10¹⁹ m⁻³ | 2.6 eV | 254.8 mm | 1.41 × 10²¹ m⁻³ | 5.86 Pa | 0.950 | 0.995 | 2.6 eV |

The mean free path falls with density until the target cools below the ionisation threshold and
then jumps to a quarter of a metre, which is the detachment signature: a cold target does not
ionise, so the neutrals cross it. The two losses come from the same coefficients, charge exchange
taking the momentum and each ionisation costing 30 eV of power, and only the power loss brings
the target down. Momentum loss on its own raises it, because thinning the target plasma leaves
a higher temperature carrying the same sheath flux.

### Incidence against the target's own surface

The component files are poloidal contours at successive toroidal angles, cut at 0.500° on
the divertor targets, which is the toroidal extent of one target element. The contour moves
up to 27 mm in R and 12 mm in Z between adjacent cuts, so the surface has a toroidal
derivative and its normal is the cross product of the two: the poloidal tangent is
(R_u, 0, Z_u) in the cylindrical basis and the toroidal one (R_φ, R, Z_φ), giving a normal
(−R Z_u, Z_u R_φ − R_u Z_φ, R R_u). That derivative is the inclination the elements are
built with; without it the normal is poloidal and the angle is the one a swept contour gives.

`python -m w7x_twin incidence` measures both against the same strikes, at 5 MW:

| Measured against | Median | 5th | 95th | Range |
|---|---|---|---|---|
| The swept contour | 17.03° | — | — | 2.09 to 24.27° |
| Its own surface | 2.60° | — | — | 0.12 to 9.85° |
| Horizontal target, own surface | 7.00° | 1.48° | 9.74° | 4 elements struck |
| Vertical target, own surface | 1.38° | 0.21° | 7.56° | 9 elements struck |

The design bound is published as up to 3°. The fan samples the strike band at five
launch planes anchored to the vacuum boundary, and its median of 2.60° is the same
angle the discharge comparison's own fan measures. 64 % of strikes arrive inside the
bound; the vertical target takes the field at a fifth of the horizontal one's angle,
and within a target the angle moves 0.09° across one element against 2.23° of scatter
between elements, so the element is the resolution the incidence needs.

The layer solved at each of the three angles:

| Incidence from | Angle | q_∥ | Width | Wetted | T_t | q_target |
|---|---|---|---|---|---|---|
| The swept contour | 17.03° | 22.2 MW/m² | 75.9 mm | 0.769 m² | 2.54 eV | 6.50 MW/m² |
| Its own surface | 2.60° | 130.3 MW/m² | 40.4 mm | 0.847 m² | 31.7 eV | 5.90 MW/m² |
| The design bound | 3.00° | 113.9 MW/m² | 42.3 mm | 0.838 m² | 26.2 eV | 5.96 MW/m² |

The measured angle and the design bound differ by one per cent in the load they imply.
The swept contour puts the target temperature an order of magnitude below either.

## Bootstrap current to strike line

The chain the machine runs on: self-generated current shifts the edge transform, which
moves the island chain the divertor is built around, which walks the strike line across
the target. `python -m w7x_twin migration` runs it end to end. For each pressure the
bootstrap current is solved to self-consistency, the field of the resulting plasma
current is added to the coil field by volume integration, field lines are traced in the
total field, and each strike is placed by arc length along the target's own poloidal
contour.

Position is arc length along the horizontal and vertical targets laid end to end, 1.025 m
of contour, since the strike line walks from one onto the other as the current rises.

| ⟨β⟩ | I_boot | ι at edge | Mean position | Displacement | Leading edge | Floor | Lines |
|---|---|---|---|---|---|---|---|
| 0 % | 0 | 0.95370 | 0.799 m | — | — | 22.1 mm | 67 |
| 0.55 % | −6.85 kA | 0.96845 | 0.853 m | +54.0 mm | +201.7 mm | 23.0 mm | 81 |
| 0.80 % | −9.93 kA | 0.97505 | 0.883 m | +83.9 mm | +247.9 mm | 23.5 mm | 60 |
| 1.05 % | −12.86 kA | 0.98084 | 0.921 m | +121.6 mm | +306.1 mm | 24.5 mm | 38 |

The wetted zone moves 54 mm along the targets at 6.9 kA and 122 mm at 12.9 kA, against a
resolution floor of 22 to 25 mm, so the displacement exceeds what the integration can
place by factors of two to five. The leading edge, the tenth percentile of the
distribution, moves three times as far as the mean: the zone both translates and
compresses, and by 12.9 kA it has crossed from the horizontal target onto the vertical
one. This is the behaviour that makes the current worth controlling, and what
counter-ECCD is used to hold in place during long pulses. Measured on the horizontal target
alone the displacement is invisible, since by 9.9 kA no line reaches that element.

The lines are launched in a narrow band just outside the last closed surface, anchored to
each case's own boundary rather than to the vacuum one, since the Shafranov shift would
otherwise slide the sampled layer between cases.

## Identified programmes

The configurations above are nominal. A programme is a discharge the machine ran, on a
date, with measurements to be held against. `programmes.py` carries fourteen
across five campaigns, taken from open publications rather than from the machine's
archive. Those publications report most quantities as figures, so each entry carries only
what the text states in numbers and records the accuracy that implies.

The confinement enhancement is measured per discharge, and it moves by nearly a factor of
two: the gas-fuelled 20180920.017 runs at 0.70 times the ISS04 scaling
through its flat-top, the post-pellet phase of 20171207.006 at 1.30, and the highest triple
product at 1.40. The model is run at each discharge's own measured value rather than at one
constant, so what a residual measures is the power balance and not the constant.

| Quantity | Model | Published | Residual |
|---|---|---|---|
| Stored energy, 20171207.006 heating phase | 1.091 MJ | 1.09 MJ | +0.1 % |
| Shafranov shift at the publication's construction | 13.9 mm | 15 mm | −7.3 % |
| Axis shift against the Minerva reconstruction | 10.2 mm | 10 mm | +1.8 % |
| Target heat flux, net of the layer radiator | 6.26 MW/m² | 6 MW/m² | +4.4 % |
| Connection length outside the island | 32.2 m | 30 m | +7.3 % |
| Carbon the measured radiated fraction needs | 0.0182 | 0.0169 ± 20 % | +8.0 % |
| Wetted area at the strike line | 1.09 m² | 1 m² | +8.5 % |
| Connection length inside the island | 211 m | 250 m | −15 % |
| Strike-line width at the measured diffusivity | 34.9 mm | 2 to 4 cm | +16 % |
| Strike-line width | 36.6 mm | 2 to 4 cm | +22 % |
| Drift-kinetic share over the peaking range, 20171207.006 | 0.22 to 0.48 | 0.45 | band, inside |
| Drift-kinetic share over the core region, 20181016.037 | 0.00 to 0.97 | 0.50 | band, inside |
| Electron share over the core region, 20181016.037 | 0.00 to 0.23 | 0.30 | band, inside |
| Density peaking of the reference profile | 1.06 | 1.2 to 2.8 | −47 % |
| Drift-kinetic share at the core point, 20171207.006 | 0.22 | 0.45 | −51 % |
| Drift-kinetic share at the core point, 20181016.037 | 0.17 | 0.50 | −65 % |
| Electron share at the core point, 20181016.037 | 0.017 | 0.30 | −94 % |
| Stored energy, 20180919.033 heating phase | 0.83 MJ | 0.30 MJ | +176 % |
| Stored energy, 20180919.033 beam phase | 1.64 MJ | 0.50 MJ | +227 % |

Thirteen of the twenty sit inside the accuracy their sources support. The share
comparisons come in two constructions because the sources state them two ways. The
region rows read the share the way the papers phrase it — a fraction of the input power
over the core region, forty to fifty per cent for the totals and twenty to forty for the
electron channel inside 0.30 m — and all three sit inside the published statements, the
drift-kinetic range over the reported peaking bracketing 20171207.006's 0.45. The
core-point rows demand the share at a single radius, which is more than the sources
claim, and there the model sits at half the published totals; the electron point value
carries in addition the pitch-angle-only collision operator, which understates the
electron transport a balance with energy scattering computes.

The pellet row runs at the measured 4.9 MW, the discharge's own measured enhancement of 1.30
and the reference profile's axis density of 8.0 × 10¹⁹ m⁻³; inverting for the density that
reproduces the measured energy gives 7.99 × 10¹⁹ m⁻³.

The neoclassical share is the drift-kinetic heat flux through 0.30 m as a fraction of the
heating, formed on the discharge's own profile the way a power balance analysis forms it.
Both Onsager drives are carried: the temperature gradient through the diffusivity and the
density gradient through the off-diagonal coefficient of the same pair, which a peaked
post-pellet profile makes comparable to the first.

For 20171207.006, at its measured 4.9 MW, peak density of 10²⁰ m⁻³, equilibrated
temperatures and effective charge of 1.5:

| Density peaking | Share of the input power | With no radial electric field |
|---|---|---|
| 1.2 | 0.484 | 1.012 |
| 2.0 | 0.320 | 0.845 |
| 2.8 | 0.222 | 0.718 |

The published 0.45 is reproduced at a peaking of 1.35, at the bottom of the span whose
top the source's own figure places the post-pellet phases at, so the row formed at the
faithful peaking reads half the published value. The field is the lever, and it is
measured: sweeping the imposed radial field at the post-pellet peaking, the published
0.45 is reproduced at −2.9 kV/m where the ambipolar root on this profile sits near −12.
The machine's own post-pellet measurements carry an electric-field well reaching
−40 kV/m at ρ of 0.7 to 0.8 with the core on the ion root, so the root's magnitude is
not the excess: what the comparison indicts is the suppression the tabulated
coefficient applies per unit field at these parameters, or the field the published
balance itself ran at.

The Shafranov shift is the largest residual. The equilibrium's own shift and the field of
the plasma currents integrated separately agree to 1.2 %, at 28 to 29 mm, where the machine
reports 1 to 2 cm. The axis responds to the pressure it sits in rather than to the volume
average, so holding ⟨β⟩ at 1.05 % and varying the profile as
(1 − s)^α:

| α | Pressure peaking | Peak pressure | ⟨β⟩ | Shift |
|---|---|---|---|---|
| 0.5 | 1.50 | 36.1 kPa | 1.048 % | 16.21 mm |
| 1.0 | 2.00 | 48.2 kPa | 1.053 % | 21.25 mm |
| 1.5 | 2.50 | 60.3 kPa | 1.057 % | 26.20 mm |
| 2.0 | 3.00 | 72.4 kPa | 1.060 % | 31.04 mm |
| 2.5 | 3.50 | 83.4 kPa | 1.050 % | 35.33 mm |
| 3.0 | 4.00 | 95.1 kPa | 1.050 % | 39.78 mm |

The shift runs 16 to 40 mm at one β, so the profile shape is what carries the residual. The
package's kinetic profile has a pressure peaking of 2.84, which the scan places at 29 mm, and
the published 1 to 2 cm is reached at a peaking of 1.5. Its density peaking is 1.06 against a
published 1.2 to 2.8, so the pressure peaking of 2.84 is carried by the temperature profile
and not by the density.

On the package's own profiles, where the pressure is n(T_e + T_i) and the two shapes are set
separately, the shift follows the pressure peaking and not the family behind it: a peaking of
1.99 from a flat density and a linear temperature gives 20.87 mm against the 21.25 mm the
analytic family gives at 2.00.

| Density peaking | T exponent | Pressure peaking | Shift |
|---|---|---|---|
| 1.06 | 1 | 1.99 | 20.87 mm |
| 1.06 | 2 | 2.84 | 29.12 mm |
| 1.20 | 1 | 2.08 | 21.86 mm |
| 1.20 | 2 | 2.93 | 30.09 mm |
| 2.00 | 1 | 2.78 | 27.55 mm |
| 2.00 | 2 | 3.77 | 36.45 mm |
| 2.80 | 1 | 3.66 | 31.23 mm |
| 2.80 | 2 | 4.89 | 40.46 mm |

Raising the density peaking into the published 1.2 to 2.8 raises the pressure peaking with it,
so the shift moves away from 1 to 2 cm. The flattest profile the family reaches gives 20.87 mm
at a density peaking of 1.06, below the published range, so the two published quantities do
not hold together at ⟨β⟩ = 1.05 % here: the peaking is reported for post-pellet phases and the
shift for the machine generally.

The digitised profiles put the same question from the measured side. The post-pellet
density and both temperatures of 20181016.037 build a pressure whose peaking is 3.05, and
the equilibrium at that pressure shifts 37.8 mm as measured and 32.4 mm rescaled to
⟨β⟩ = 1.05 %: the machine's own profiles sit at the steep end of the family, so the
residual is not the synthetic shape's doing.

The equilibrium current does not close it either, and the coupled solve measures that:
with the bootstrap carried self-consistently the shift is 28.35 mm at ⟨β⟩ = 1.05 % from
the current solve alone and 31.09 mm at the coupled 1.33 %, above the pressure-only
21.6 mm at the same beta. The current raises the edge transform and the shift together.

The frame and the construction close it. A free-boundary equilibrium moves its boundary
at finite beta, so the axis is measured against the boundary's own m = 0 centre, and the
published figures are reproduced by their own constructions. The publication's 1 to 2 cm
was drawn from a VMEC scan at the standard p ∝ (1 − s) profile overlaid on a 3 cm
tomogram grid: that construction, the standard family at a pressure peaking of 2 rescaled
to one per cent beta, shifts 13.9 mm in the boundary frame against the stated 15 mm,
−7.3 %. The Minerva reconstruction of XP_20171108.040 at 346 kJ reports the axis moving
order 1 cm at its own beta of 0.38 %: the drawn gas-fuelled temperatures over the
flat-topped density at that beta move the axis 10.2 mm in the boundary frame, +1.8 %.
Both rows sit inside their sources' accuracy; the steeper drawn post-pellet pressure at a
peaking of 3.05 shifts 26.1 mm at one per cent, which is the profile the 1 to 2 cm was
never drawn for.

The two 20180919.033 stored energies are forward solves at the reference profile's density,
which is not this discharge's: its density is published as a figure. Inverting, the measured
0.30 MJ at 2 MW requires an axis density of 1.16 × 10¹⁹ m⁻³ and the measured 0.50 MJ at
3.4 MW requires 2.08 × 10¹⁹ m⁻³. The first is the right order for a 2 MW electron-cyclotron
phase. The second is not: the source states the central density passed 10²⁰ m⁻³ during the
beam phase, five times what the model needs to hold the stored energy down to what was
measured, so the model over-predicts confinement at high density and peaked profiles.

`programmes.GYROKINETIC_BENCHMARKS` carries the nonlinear gyrokinetic heat fluxes another
code published for 20181016.037, which are a second treatment of that discharge rather than
a measurement of it: 0.42 ± 0.05 MW into the ions and 0.49 ± 0.07 MW into the electrons at
ρ = 0.4, and 3.34 ± 0.11 against 0.80 ± 0.02 at ρ = 0.8 in a flux tube, falling to
1.77 ± 0.08 and 0.30 ± 0.02 when the same case is solved radially globally.

## An equilibrium that carries its own island

VMEC assumes nested flux surfaces, so everything this package says about islands is said
about the traced field. `stepped_pressure.py` builds a SPEC input from a converged VMEC
solution, which drops that assumption: volumes separated by ideal interfaces, a Beltrami
field in each, and islands wherever a volume's own transform crosses a rational.

The configuration solved is one whose transform crosses 5/6 inside the plasma, at s = 0.28,
so the resonance is interior and sits inside a volume rather than on an interface. The same
boundary, pressure and transform profile drive both codes.

The island is measured the same way in each. A field line inside an island of a chain still
encircles the magnetic axis, because the chain wraps around it, so the signature is not
libration but a plateau in the winding number, locked to the resonant value across the
island's width.

In the traced vacuum field the 5/6 island spans 17.4 mm over 15 of 90 densely sampled
trajectories, whose winding implies a transform locked to 0.83333.

The interior interfaces are started at the VMEC surfaces of the same flux, written into the
input as the geometry SPEC reads when `Linitialize` is zero; its own initialisation carries
the boundary inwards as ψ^(m/2) and floors an order of magnitude higher.

Convergence to SPEC's own tolerance is governed by two placement laws, both measured here.
An ideal interface can only exist on a strongly irrational surface: an interface whose
transform lands within about 0.25/q² of a rational p/q has no equilibrium to converge to,
and the Newton stalls near 10⁻³ however the resolution is raised. And a sliver volume
stalls the solve the same way, so the placement keeps every interface pair at least half
the even spacing apart. The spectral-condensation weight must also be on: without it the
interfaces' tangential freedom leaves the Jacobian singular, worth a factor of 25 alone.
Interfaces are therefore placed at noble transforms, clear of every rational below
denominator 24, with the pair around 5/6 straddling it.

So placed, the bracketing case converges to machine precision at the physics pressure:

| Volumes | Placement | Force residual | Island at 5/6 |
|---|---|---|---|
| 4 | bracketing | 1.67 × 10⁻¹⁵ | 27.2 mm |
| 8 | bracketing | 8.76 × 10⁻⁵ | 23.7 mm |
| 12 | on the resonance | 2.65 × 10⁻⁴ | 6.6 mm |

The four-volume bracketing equilibrium carries its island at a force residual five orders
below the 10⁻¹⁰ tolerance, against the traced vacuum field's 17.4 mm at the same
resonance. On the resonance the island closes by construction, which is the same
equilibrium asked a different question, and the record keeps both placements at every
volume count: `python -m w7x_twin spec`, `results/equilibrium/spec.json`.

## Deposited energy as machine state

What a machine carries between programmes is the history of what has been deposited on its
walls. `machine.py` accumulates deposited energy per plasma-facing element, keyed by
element, module and unit, with the geometry version and the in-vessel epoch stamped on it, so
a record accumulated against one coil model is not continued against another's.

The fan is launched at the equivalent point of every module, which makes the per-module
spread the periodicity test: a five-fold symmetric field loads the five modules of any
element equally. The upper-to-lower ratio is a separate number, since a fan launched on the
midplane loads the upper units by its own geometry rather than by the field.

## Rendered twin

`artifact/w7x_twin3d.html` is the machine as a single self-contained page, built by
`artifact/build_twin3d.py` from two exports: the geometry the physics resolved against,
and the per-circuit vacuum field response.

The response table holds the field each circuit produces at unit current per turn, so moving
a coil current is one weighted sum over circuits. The page performs that sum, locates the
magnetic axis as the fixed point of the return map, and traces field lines by fourth-order
Runge-Kutta until they reach the vessel wall, so moving a slider moves the islands, the
strike points and the rotational transform. The transform is measured from the winding angle
about the co-traced axis, the same way `fieldlines.py` measures it.

The page carries two coarsenings against the model, and each has a measured cost. Its field
response keeps all 36 toroidal planes per field period and halves each poloidal direction to
61 × 61 from the 121 × 121 × 36 grid the model solves on, and its tracer runs 48 steps per
field period with a wall test every eighth step against the model's 120 steps and a test at
every one.

| Grid | Tracer | Axis R | Against the model | Winding of the outermost line | Strike points move |
|---|---|---|---|---|---|
| model | model | 5.947425 m | — | 1.040063 | — |
| model | page | 5.947407 m | −0.018 mm | 1.039758 | 134 mm |
| page | model | 5.947471 m | +0.046 mm | 1.039770 | 0.2 mm |
| page | page | 5.947470 m | +0.045 mm | 1.039624 | 134 mm |

The stored grid carries almost none of the strike error: under the model's tracer its strike
points move 0.2 mm in the median and 1.1 mm at worst, connection lengths hold to 10⁻⁵, and
the same 21 of 24 launched lines terminate. The page's tracer carries nearly all of it:
134 mm in the median, 1.3 m at worst, connection lengths changing 7 %, and the eighth-step
wall test strikes the three lines the model keeps circulating. The median sits at the scale
of the 122 mm the strike line moves at 12.9 kA of bootstrap current, so the page's exhaust
is qualitative at that scale; a per-line strike point wants the model's tracer.

Flux surfaces are traced: one field line per surface followed for
forty-eight turns, its crossings of sixty toroidal planes sorted by poloidal angle about the
co-traced axis, resampled onto a uniform angle grid and closed into a tube.

Beyond the seven superconducting circuits, two things are drivable. A pressure slider scales
one plasma-current block computed at ⟨β⟩ = 1.05 %, which is linear in β to 2 % over the
range this model runs. The trim and control circuits sit on a 31 × 31 × 45 grid over the
whole torus with no symmetry assumed. The four type A trim coils are one winding at 72°
intervals, so one stored block serves all four under a rotation of the toroidal index, which
the exporter verifies against the directly computed circuits to 1.5 × 10⁻¹¹.

`artifact/check_page_numerics.py` runs the page's own field contraction, Runge-Kutta step
and axis search under node against the exported bundle. The axis agrees with the Python
tracer on the same grid to the micrometre and the winding numbers to 1.4 × 10⁻⁷ per line
across the fan; the trim circuits leave n = 1 to 4 at 2 × 10⁻⁹ T unpowered against 0.145 T
at n = 5, and one energised trim coil puts 1.477 mT into n = 1.

Around that: eighty-five coil filaments coloured by circuit, the plasma vessel, the divertor
and baffles as a hundred instances each carrying its module and unit, and a wedge of the
torus that cuts away. A readout pane binds the verification record, each number beside its
band and the command that produced it. A sound toggle derives two tones from the traced
state, the base pitch following the field on the traced axis and the second sitting at the
edge transform times it, so a locked island chain is audible as consonance; it is labelled
derived, not recorded. The page states the geometry version it was built from.

```bash
./run.sh -m w7x_twin export-geometry      # coils, vessel, components, traced lines
./run.sh -m w7x_twin export-field         # per-circuit, plasma and auxiliary response
python artifact/build_twin3d.py
python artifact/check_page_numerics.py    # the page's tracer against the model's
```
