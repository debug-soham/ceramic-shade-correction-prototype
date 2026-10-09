# Shade-correction prototype (synthetic data)

The case study makes one central claim: a print-file correction can be computed from spectrophotometer readings at a few spots plus the file's ink values at those spots, without an accurate model of the kiln. This prototype tests that claim. It also shows where the approach breaks: firing noise, the limits of whole-file edits, and metamerism.

```
python shade_correction_sim.py     # numpy + scipy, about 90 s; writes results.txt
```

**What is simulated**

- **Inks and colour**: six ceramic inks (blue, brown, beige, yellow, pink, black) mixed by Kubelka-Munk over a white engobe, converted to CIE L\*a\*b\* and compared with ΔE2000. The ΔE2000 code is checked against a published reference pair.
- **The kiln, day to day**: changes ink strength, tone response, pigment hue and the base colour. A simple % edit therefore cannot undo it exactly, which is realistic.
- **Physical noise**: tile-to-tile firing variation and spectro measurement noise.
- **The correction engine**:
  - starts from a sensitivity model built on a *different* day (stale);
  - solves the smallest bounded edit that brings all spots into tolerance;
  - prints 3 bracketing variants (100/60/140%);
  - refits the sensitivity from the measured tiles if another round is needed.

**What it is not**: a forecast of real plant numbers. It contains no human baseline and no image processing. The physics is deliberately simple. Real numbers come from the client's tiles in Phase 0.

Results are summarised in Appendix A of `../case-study.md`.
