# Outstanding work

1. Converge the saturation response: every anchor between a/L_T = 1.5 and 3.0 stands in a
   box reaching ky_max 0.333, and the inner surfaces are still falling at 1.53.
2. Reproduce the measured confinement: 0.70 times ISS04 gas-fuelled, 1.86 between regimes,
   profiles inside their bands.
3. Regenerate the profile-residual, computed-balance and turbulence-dependent discharge
   records once the computed transport channels are calibrated.
4. Extend the saturation response, the drift-kinetic scan and the stepped-pressure solves to
   the remaining configurations of the library.
5. Resolve the scrape-off layer with a three-dimensional edge transport model carrying
   neutrals and impurities beyond the two-point and Lengyel chain.
6. Substitute IPP's measured coil database for the reconstructed trim and control paths and
   the as-designed superconducting filaments.
7. Calibrate against archived waveforms and measured profiles, benchmark against
   reconstructed equilibria, validate forward-modelled diagnostic signals against archived
   ones, and predict a withheld campaign, reproducing the discharges whose parameters the
   publications do not state in numbers.
8. Regenerate the records that predate the geometry they are read under. `w7x-twin
   records` puts 15 of 36 on the current inputs: eight carry component contours from
   before `cut-contours` rewrote them, most predate the constructed-winding part, and
   `gyrokinetic.json` predates the epoch. Three records whose commands now stamp the
   inputs they read were written before that stamping and carry no digests.
9. Produce the shear-quench record from a command. `results/turbulence/shear_quench.json`
   supplies the 0.56 that `transport.SHEAR_QUENCH_ALPHA` carries and the imposed-shear
   runs behind it, and nothing in this package writes it.
10. Produce the connection-length figure from a command. The README displays
    `docs/w7x_connection_length_standard.png` and nothing here renders it.
