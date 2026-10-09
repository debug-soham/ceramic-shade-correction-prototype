"""
Shade-correction prototype on synthetic data (numpy + scipy only).

What it is meant to show (and nothing more):
  1. Given spectrophotometer readings of the master tile and the current
     production tile at a few known spots, plus the print file's ink values at
     those spots, you can compute a per-channel correction to the print file.
     You do NOT need an accurate absolute model of the kiln. A rough,
     out-of-date sensitivity model (how much each ink channel moves L*a*b*)
     is enough, because the correction only needs the slope, and the slope is
     refined from the tiles that come out of each firing.
  2. Whether any file correction can work depends on the firing
     repeatability (the "noise floor"). If two tiles printed from the same
     file differ by more than the tolerance, no software fixes that. This is
     why the plan measures repeatability before promising anything.
  3. Matching under one light (D65) does not guarantee matching under another
     (incandescent A). The tool should check both.

The physics is deliberately simple: Kubelka-Munk mixing of six ceramic-ink
absorption spectra over a white engobe/glaze, analytic CIE colour-matching
functions, and blackbody approximations of D65 and A. The numbers are
illustrative. The real numbers come from the client's own tiles.

Run:  python shade_correction_sim.py          (about a minute)
"""

import numpy as np
from scipy.optimize import lsq_linear, minimize

# ----------------------------------------------------------------------------
# Colour science
# ----------------------------------------------------------------------------
WL = np.arange(400, 701, 10).astype(float)


def _lobe(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


# Wyman, Sloan & Shirley (2013) analytic fit of the CIE 1931 2-degree observer.
XBAR = 1.056 * _lobe(WL, 599.8, 37.9, 31.0) + 0.362 * _lobe(WL, 442.0, 16.0, 26.7) - 0.065 * _lobe(WL, 501.1, 20.4, 26.2)
YBAR = 0.821 * _lobe(WL, 568.8, 46.9, 40.5) + 0.286 * _lobe(WL, 530.9, 16.3, 31.1)
ZBAR = 1.217 * _lobe(WL, 437.0, 11.8, 36.0) + 0.681 * _lobe(WL, 459.0, 26.0, 13.8)


def _planck(t):
    lam = WL * 1e-9
    return 1.0 / (lam ** 5 * (np.exp(1.4388e-2 / (lam * t)) - 1.0))


ILLUMINANTS = {"D65": _planck(6504.0), "A": _planck(2856.0)}  # blackbody approximations


def to_lab(refl, ill="D65"):
    s = ILLUMINANTS[ill]
    k = 100.0 / np.sum(s * YBAR)
    xyz = np.stack([k * np.sum(s * cmf * refl, axis=-1) for cmf in (XBAR, YBAR, ZBAR)], axis=-1)
    white = np.array([k * np.sum(s * cmf) for cmf in (XBAR, YBAR, ZBAR)])
    t = xyz / white
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def de2000(lab1, lab2):
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, float), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, float), -1, 0)
    cbar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(cbar ** 7 / (cbar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dlp, dcp = L2 - L1, c2p - c1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(c1p * c2p == 0, 0, dhp)
    dHp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp / 2))
    lbp, cbp, hsum = (L1 + L2) / 2, (c1p + c2p) / 2, h1p + h2p
    hbp = np.where(c1p * c2p == 0, hsum,
                   np.where(np.abs(h1p - h2p) <= 180, hsum / 2,
                            np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    t = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    rc = 2 * np.sqrt(cbp ** 7 / (cbp ** 7 + 25.0 ** 7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / np.sqrt(20 + (lbp - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * t
    rt = -np.sin(np.radians(2 * dtheta)) * rc
    return np.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dHp / sh) ** 2 + rt * (dcp / sc) * (dHp / sh))


# Sharma et al. (2005) reference pair, to make sure the formula is right.
assert abs(de2000([50, 2.6772, -79.7751], [50, 0, -82.7485]) - 2.0425) < 1e-3

# ----------------------------------------------------------------------------
# Synthetic tile: six ceramic ink channels, Kubelka-Munk over white engobe
# ----------------------------------------------------------------------------
INKS = ["blue", "brown", "beige", "yellow", "pink", "black"]


def _band(mu, sig):
    return np.exp(-0.5 * ((WL - mu) / sig) ** 2)


KS_INK = np.array([
    0.15 + 2.2 * _band(600, 55),             # blue: absorbs orange/red
    0.45 + 2.4 * np.exp(-(WL - 400) / 90),   # brown: broad, strong at short wavelengths
    0.03 + 0.55 * np.exp(-(WL - 400) / 70),  # beige: weak short-wave absorption
    0.02 + 2.8 * _band(430, 28),             # yellow: absorbs blue
    0.05 + 1.8 * _band(535, 40),             # pink: absorbs green
    6.0 + 0 * WL,                            # black: flat
])
KS_SUBSTRATE = 0.015 + 0.03 * np.exp(-(WL - 400) / 60)
GAMMA = np.array([1.15, 1.10, 1.00, 1.05, 1.10, 1.20])  # tone response of each channel

# A stone-look SKU, measured at four spots. Rows: spots. Columns: ink coverage 0..1
# taken from the print file at the spot location.
BASE_SPOTS = np.array([
    [0.02, 0.10, 0.35, 0.12, 0.05, 0.00],  # background
    [0.05, 0.35, 0.30, 0.10, 0.08, 0.03],  # mid vein
    [0.08, 0.55, 0.20, 0.05, 0.10, 0.12],  # dark vein
    [0.20, 0.15, 0.20, 0.03, 0.04, 0.06],  # cool accent
])


def nominal_state():
    return {"s": np.ones(6), "gam": np.ones(6), "shift": np.zeros(6), "sub": 0.0, "sub_y": 0.0}


def drifted_state(rng, scale=1.0):
    """Process state on a given day: kiln temperature, ink lot, glaze batch, body.
    Drift changes ink strength, tone response, pigment hue (band position) and the base colour,
    so a simple per-channel gain cannot undo it exactly."""
    kiln = rng.normal() * scale
    s = 1 + scale * rng.normal(0, 0.08, 6)                                 # ink lot / glaze, per ink
    s = s * (1 + kiln * np.array([0.04, 0.10, 0.05, -0.04, -0.10, 0.05]))  # common kiln factor
    gam = 1 + scale * rng.normal(0, 0.05, 6)                               # tone response / drop size
    shift = kiln * np.array([0, 4, 3, 3, -5, 0]) + scale * rng.normal(0, 3, 6)  # pigment hue, nm
    return {"s": s, "gam": gam, "shift": shift,
            "sub": scale * rng.normal(0, 0.3), "sub_y": scale * (0.015 * kiln + rng.normal(0, 0.008))}


def _ink_ks(shift):
    return np.array([np.interp(WL - shift[i], WL, KS_INK[i]) for i in range(6)])


def fire(c, state):
    """Ink coverage (..., 6) -> fired diffuse reflectance (..., n_wavelengths)."""
    eff = state["s"] * np.clip(c, 0, 1) ** (GAMMA * state["gam"])
    ks = KS_SUBSTRATE * (1 + state["sub"]) + state["sub_y"] * np.exp(-(WL - 400) / 60) + eff @ _ink_ks(state["shift"])
    ks = np.maximum(ks, 1e-6)
    return 1 + ks - np.sqrt(ks ** 2 + 2 * ks)


def tile_noise(state, rng, sigma):
    """One physical tile: the day's process state plus tile-to-tile firing variation."""
    common = rng.normal(0, sigma)
    return dict(state, s=state["s"] * (1 + common + rng.normal(0, sigma / 2, 6)),
                sub=state["sub"] + rng.normal(0, sigma * 2))


def apply_gains(spots, g):
    """Edit to the print file. 6 values: one flat percentage per channel (what QC does today,
    "brown +5%"). 12 values: a two-point tone curve per channel (gain in light areas, gain in
    dark areas), i.e. a Photoshop Curves adjustment per channel."""
    if len(g) == 6:
        return np.clip(spots * g, 0, 1)
    w = np.clip((spots - 0.1) / 0.4, 0, 1)
    return np.clip(spots * (g[:6] + (g[6:] - g[:6]) * w), 0, 1)


MEAS_NOISE = 0.08  # L*a*b* units, averaged spectro reading


def print_fire_measure(spots, g, state, rng, sigma_rep):
    lab = to_lab(fire(apply_gains(spots, g), tile_noise(state, rng, sigma_rep)))
    return lab + rng.normal(0, MEAS_NOISE, lab.shape)


# ----------------------------------------------------------------------------
# Correction engine
# ----------------------------------------------------------------------------
G_LO, G_HI = 0.6, 1.6  # allowed per-channel gain range on the print file
LAMBDA = 4.0           # prefers small edits (minimum-change solution)


def jacobian(state, spots, g, h=0.02):
    base = to_lab(fire(apply_gains(spots, g), state)).ravel()
    J = np.zeros((base.size, len(g)))
    for i in range(len(g)):
        g2 = g.copy()
        g2[i] += h
        J[:, i] = (to_lab(fire(apply_gains(spots, g2), state)).ravel() - base) / h
    return J


def solve_step(J, resid, g):
    A = np.vstack([J, np.sqrt(LAMBDA) * np.eye(len(g))])
    b = np.concatenate([resid, np.zeros(len(g))])
    return lsq_linear(A, b, bounds=(G_LO - g, G_HI - g)).x


def refit_jacobian(J_prior, X, Y, mu=0.02):
    """Pull the sensitivity toward what the fired tiles actually showed (ridge to the prior)."""
    X, Y = np.asarray(X).T, np.asarray(Y).T  # (n_params, n), (n_spots*3, n)
    return (Y @ X.T + mu * J_prior) @ np.linalg.inv(X @ X.T + mu * np.eye(X.shape[0]))


def run_correction(spots, target, true_state, model_state, rng, sigma_rep, tol=1.0,
                   variants=(1.0, 0.6, 1.4), max_rounds=4, log=None, n_params=6):
    """Returns rounds needed (0 = first sample already fine, max_rounds+1 = gave up)."""
    g = np.ones(n_params)
    m = print_fire_measure(spots, g, true_state, rng, sigma_rep)  # first sample: happens today anyway
    if de2000(target, m).max() <= tol:
        return 0, g, m, None
    J = jacobian(model_state, spots, g)
    points = [(g, m)]
    first_pred_err = None
    for r in range(1, max_rounds + 1):
        dg = solve_step(J, (target - m).ravel(), g)
        cands = [np.clip(g + a * dg, G_LO, G_HI) for a in variants]
        meas = [print_fire_measure(spots, c, true_state, rng, sigma_rep) for c in cands]
        if r == 1:
            pred = m + (J @ dg).reshape(m.shape)
            first_pred_err = float(np.median(de2000(pred, meas[0])))
        if log is not None:
            log.append((r, g.copy(), m.copy(), dg.copy(), cands, meas))
        scores = [de2000(target, x).max() for x in meas]
        best = int(np.argmin(scores))
        points += list(zip(cands, meas))
        if scores[best] <= tol:
            return r, cands[best], meas[best], first_pred_err
        g, m = cands[best], meas[best]
        others = [(pg, pm) for pg, pm in points if not np.allclose(pg, g)]
        J = refit_jacobian(J, [pg - g for pg, _ in others], [(pm - m).ravel() for _, pm in others])
    return max_rounds + 1, g, m, first_pred_err


def best_achievable(spots, target, true_state, n_params=6):
    """With perfect knowledge of today's process, how close can the file edit get?"""
    def f(g):
        return np.mean(de2000(target, to_lab(fire(apply_gains(spots, g), true_state))) ** 2)
    res = minimize(f, np.ones(n_params), method="L-BFGS-B", bounds=[(G_LO, G_HI)] * n_params)
    return de2000(target, to_lab(fire(apply_gains(spots, res.x), true_state))).max()


def make_scenario(rng, prior_scale=1.0, drift_scale=1.0):
    spots = np.clip(BASE_SPOTS * rng.uniform(0.75, 1.25, BASE_SPOTS.shape), 0, 1)
    master_state = nominal_state()                    # process on the day the master was approved
    target = to_lab(fire(spots, master_state))        # what the spectro reads on the master tile
    today = drifted_state(rng, drift_scale)           # process today (unknown to us)
    prior = drifted_state(rng, prior_scale)           # characterisation chart fired on some other day
    return spots, target, today, prior


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def summarise(rounds, max_rounds=4):
    r = np.asarray(rounds)
    return {"n": len(r), "r1": np.mean(r <= 1), "r2": np.mean(r <= 2),
            "fail": np.mean(r > max_rounds), "mean": np.mean(np.minimum(r, max_rounds + 1))}


def repeat_spread(rng, sigma_rep, n=200):
    """Typical ΔE00 between two tiles printed from the same file on the same day."""
    out = []
    for _ in range(n):
        spots, _, today, _ = make_scenario(rng)
        a = print_fire_measure(spots, np.ones(6), today, rng, sigma_rep)
        b = print_fire_measure(spots, np.ones(6), today, rng, sigma_rep)
        out.append(de2000(a, b).max())
    return np.median(out), np.percentile(out, 90)


def collect(rng, n, sigma_rep, prior_scale=1.0, variants=(1.0, 0.6, 1.4), oracle=False, n_params=6):
    rounds, pred_errs, meta_A, initial = [], [], [], []
    while len(rounds) < n:
        spots, target, today, prior = make_scenario(rng, prior_scale)
        init = de2000(target, to_lab(fire(spots, today))).max()
        if init <= 1.0:
            continue  # only batches that need correcting
        model = today if oracle else prior
        r, g, _, pe = run_correction(spots, target, today, model, rng, sigma_rep, variants=variants,
                                     n_params=n_params)
        rounds.append(r)
        initial.append(init)
        if pe is not None:
            pred_errs.append(pe)
        if r <= 4:
            master_A = to_lab(fire(spots, nominal_state()), "A")
            prod_A = to_lab(fire(apply_gains(spots, g), today), "A")
            meta_A.append(de2000(master_A, prod_A).max())
    return summarise(rounds), np.array(pred_errs), np.array(meta_A), np.array(initial)


def worked_example(rng, sigma_rep):
    while True:
        spots, target, today, prior = make_scenario(rng)
        if 2.0 < de2000(target, to_lab(fire(spots, today))).max() < 3.5:
            break
    log = []
    r, g, m, _ = run_correction(spots, target, today, prior, rng, sigma_rep, log=log)
    names = ["background", "mid vein", "dark vein", "accent"]
    lines = ["Worked example (one synthetic SKU, four measured spots)", ""]
    _, g0, m0, dg, cands, meas = log[0]
    lines.append("First production sample vs master, before any correction:")
    lines.append(f"  {'spot':<11}{'dL':>7}{'da':>7}{'db':>7}{'dE00':>7}")
    for k, nm in enumerate(names):
        d = m0[k] - target[k]
        lines.append(f"  {nm:<11}{d[0]:7.2f}{d[1]:7.2f}{d[2]:7.2f}{de2000(target[k], m0[k]):7.2f}")
    lines.append("")
    lines.append("Recommended change to the print file (per channel, relative):")
    lines.append("  " + ", ".join(f"{INKS[i]} {100 * dg[i]:+.1f}%" for i in range(6) if abs(dg[i]) >= 0.005))
    lines.append("Variants to print in round 1: 100%, 60% and 140% of that change.")
    lines.append("")
    lines.append("Round 1 result (measured):")
    for a, mm in zip((1.0, 0.6, 1.4), meas):
        lines.append(f"  {int(a * 100):>3}% of change: max dE00 = {de2000(target, mm).max():.2f}, "
                     f"mean dE00 = {de2000(target, mm).mean():.2f}")
    lines.append("")
    lines.append(f"Approved after {r} correction round(s); final per-spot dE00 = "
                 + ", ".join(f"{x:.2f}" for x in de2000(target, m)))
    return "\n".join(lines)


def main():
    rng = np.random.default_rng(7)
    N = 200
    SIGMA = 0.012  # tile-to-tile firing variation used as the default case
    out = []

    def p(s=""):
        print(s)
        out.append(s)

    p("=" * 78)
    p("Shade-correction prototype, synthetic data. Tolerance: every spot dE00 <= 1.0")
    p("=" * 78)
    p()
    p(worked_example(np.random.default_rng(3), SIGMA))
    p()

    med, p90 = repeat_spread(np.random.default_rng(11), SIGMA)
    p(f"Firing repeatability at default noise: same file, two tiles -> max-spot dE00 median {med:.2f}, P90 {p90:.2f}")
    p()

    p("1) Strategy comparison (batches that start out of tolerance, n=%d each)" % N)
    p(f"   {'strategy':<52}{'<=1 rnd':>8}{'<=2 rnd':>8}{'>4 rnd':>8}{'mean':>6}")
    rows = [
        ("Stale sensitivity model, 1 variant per round", dict(variants=(1.0,))),
        ("Stale sensitivity model, 3 variants per round", dict()),
        ("Perfect model of today's kiln, 1 variant (bound)", dict(variants=(1.0,), oracle=True)),
        ("Stale model, 3 variants, tone curve per channel", dict(n_params=12)),
    ]
    detail = None
    for name, kw in rows:
        s, pe, ma, init = collect(np.random.default_rng(101), N, SIGMA, **kw)
        if detail is None:
            detail = (pe, ma, init)
        p(f"   {name:<52}{s['r1']:8.0%}{s['r2']:8.0%}{s['fail']:8.0%}{s['mean']:6.2f}")
    pe, ma, init = detail
    p(f"   Starting error of these batches: max-spot dE00 median {np.median(init):.2f}, P90 {np.percentile(init, 90):.2f}")
    p(f"   Round-1 prediction error (predicted vs measured Lab), median dE00: {np.median(pe):.2f}")
    p(f"   After approval under D65, mismatch under incandescent A: median {np.median(ma):.2f}, "
      f"P90 {np.percentile(ma, 90):.2f}, share > 1.0: {np.mean(ma > 1.0):.0%}")
    p()

    p("2) How stale can the sensitivity model be? (3 variants per round)")
    p(f"   {'drift of model vs today':<52}{'<=1 rnd':>8}{'<=2 rnd':>8}{'>4 rnd':>8}")
    for ps in (0.0, 1.0, 2.0, 3.0):
        s, *_ = collect(np.random.default_rng(202), N, SIGMA, prior_scale=ps)
        p(f"   {f'{ps:.0f}x a typical day-to-day drift':<52}{s['r1']:8.0%}{s['r2']:8.0%}{s['fail']:8.0%}")
    p()

    p("3) Noise floor: firing repeatability decides whether any file correction can work")
    p(f"   {'tile-to-tile variation':<24}{'repeat dE00 (med/P90)':>24}{'<=1 rnd':>8}{'<=2 rnd':>8}{'>4 rnd':>8}")
    for mult in (0.5, 1.0, 2.0, 3.0, 4.0):
        sg = SIGMA * mult
        med, p90 = repeat_spread(np.random.default_rng(11), sg)
        s, *_ = collect(np.random.default_rng(303), N, sg)
        p(f"   {f'{mult:.1f}x default':<24}{f'{med:.2f} / {p90:.2f}':>24}{s['r1']:8.0%}{s['r2']:8.0%}{s['fail']:8.0%}")
    p()

    p("4) Can a whole-file edit fix a multi-colour pattern at all? (perfect knowledge, no noise)")
    for label, npar in (("flat % per channel (6 numbers)", 6), ("tone curve per channel (12 numbers)", 12)):
        rng4 = np.random.default_rng(404)
        ach = []
        while len(ach) < 100:
            spots, target, today, _ = make_scenario(rng4)
            if de2000(target, to_lab(fire(spots, today))).max() > 1.0:
                ach.append(best_achievable(spots, target, today, npar))
        ach = np.array(ach)
        p(f"   {label:<38} best max-spot dE00 median {np.median(ach):.2f}, P90 {np.percentile(ach, 90):.2f}; "
          f"cannot reach 1.0: {np.mean(ach > 1.0):.0%}")
    p()

    with open(__file__.replace("shade_correction_sim.py", "results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
