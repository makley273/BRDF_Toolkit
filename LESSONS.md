# Lessons Learned

Short notes on non-obvious problems and their fixes. One entry per issue.

---

## LM fit collapses on shiny materials (e.g. gold-metallic-paint)

**Symptom:** Fit returns `roughness` pinned to lower bound, `F0 ≈ 0`, only a weak
diffuse albedo. Rendered result looks nothing like the measurement.

**Cause:** Two compounding issues:
1. Sampling `(wi, wo)` uniformly on the hemisphere almost never hits a sharp
   specular lobe — the optimizer sees ~flat data and fits a weak Lambertian.
2. Linear-space residuals are dominated by the few specular-peak samples,
   which are too noisy (too few) to pull the optimizer toward the true peak.

**Fix:** In `LevenbergMarquardt` (`src/optimizers/levenberg_marquardt.jl`):
- Sample in Rusinkiewicz `(θ_h, φ_h, θ_d, φ_d)` space with `θ_h` biased toward
  0 — puts dense samples *on* the specular lobe.
- Fit `log1p(f)` instead of raw `f` — compresses the peak/diffuse dynamic range.

The logarithmic loss is still the default (`loss=:log`). The sampling default has
since moved on: the thesis evaluation found cosine-weighted uniform sampling beat
Rusinkiewicz warping on 294 of 312 materials by SSIM, so `sampling=:uniform_cosine`
is now the default — ridge-concentrated sampling fixed this failure, but it is not
the best general choice. For pure metals, also prefer
`GGXSpecularModel`/`CookTorranceSpecularModel` over the diffuse-plus-specular variants.
