# Shade correction for digital ceramic tile printing: prototype

A small simulation behind an FDE case study. In this case study, the fired shade of a digitally printed ceramic tile drifts away from the master tile when an SKU is run again. The proposed fix is to calculate the print-file correction instead of guessing it by eye.

**The claim being tested:** you can calculate the right correction to each ink channel without an accurate model of the kiln. You need two inputs:
- spectrophotometer readings at a few spots on the master tile and the production tile;
- a rough idea of how each ink moves the fired colour.

## What the prototype shows

The simulation runs on 200 synthetic out-of-shade batches. It mixes six ceramic inks, adds day-to-day kiln drift and tile-to-tile firing noise, and runs the proposed correction loop on each batch: the line prints 3 calculated variants per round, and the model refits from every fired tile.

| Result (synthetic) | Value |
|---|---|
| Brought into tolerance (ΔE00 ≤ 1.0 at every spot) after 1 round | 80% |
| Within 2 rounds | 90% |
| Out-of-date sensitivity model vs. a perfect one | Same result. The correction only needs the slope |

It also shows where the approach breaks:
- **Firing noise:** if repeat firings of the same file vary too much, no file correction holds.
- **Whole-file edits:** about 11% of drifts can't be fixed by a flat % edit per channel; per-channel tone curves cut that to about 5%.
- **Lighting:** a daylight match can fail under shop lighting (metamerism).

The data is synthetic and the physics is deliberately simple. The numbers show the logic works; they don't forecast real plant results. Details are in [`prototype/README.md`](prototype/README.md) and [`prototype/results.txt`](prototype/results.txt).

## Run it

```
cd prototype
python shade_correction_sim.py    # needs numpy + scipy; about 90 s
```
