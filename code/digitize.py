"""Recover the trade-level price record from Plott & Sunder (1982) Figures 2-6.

The published paper prints only the per-period AVERAGE transacted price, along the
bottom of each of its Figures 2-6.  But the figures themselves plot every single
contract: "Each dot represents one trade at the indicated price in chronological
order" (p. 679).  At the native scan resolution (3300x5300) those dots are ~12 px
across and sit on a regular horizontal grid, so the individual trades are
recoverable.

This module does the recovery.  Nothing here is fitted to the printed averages;
the averages are used only to VALIDATE (see `validate`), and every period whose
digitized mean disagrees with the printed one is flagged rather than adjusted.

Geometry of the panels, established by measurement rather than assumption:
  * y axis: major ticks every 50 francs, ~282-339 px apart depending on the
    figure's reduction.  A straight line through the ticks fits to < 0.4 franc.
  * period boundaries: full-height vertical rules, 2-4 px wide.
  * horizontal reference lines: the RE prediction (solid) and, where it differs,
    the PI prediction (dashed).  6-8 px thick and spanning the full period width.
  * dots: 11-15 px tall, on an x grid whose pitch is constant within a figure.
    A run of trades at one price merges into a single thick horizontal blob whose
    width divided by the grid pitch recovers the number of trades.

The dot/line discrimination is by THICKNESS, not by position: a reference line is
6-8 px thick everywhere across the period, a run of dots is 11-15 px.  That is what
lets a flat run of trades sitting a few francs from the RE line (market 5 periods
10 and 12) be kept while the RE line itself is removed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# ---------------------------------------------------------------- figure inventory

# Period count and the top y-axis label, both read off the scans and cross-checked
# against ps1982.markets.
N_PERIODS = {1: 11, 2: 11, 3: 12, 4: 14, 5: 13}
TOP_TICK = {1: 400, 2: 400, 3: 450, 4: 400, 5: 350}
# Figures 5 and 6 (markets 4 and 5) are printed rotated on the page.
ROTATE = {4: -90, 5: -90}
# Threshold on the longest vertical ink run that isolates exactly N_PERIODS+1
# separators in each figure.  Chosen per figure because the panels differ in
# height and in how far the separators are drawn.
SEP_RUN_MIN = {1: 1000, 2: 1200, 3: 900, 4: 1200, 5: 800}

# Each panel carries one horizontal arrow annotating the information condition
# ("<- NO INFORMATION -><- PRIVATE INFORMATION TO SIX INSIDERS ->").  It runs the
# full width of the figure at a level that has nothing to do with prices, so it
# cannot be told from a flat run of trades by geometry alone.  Its level in francs
# was read off each scan and is recorded here; the caption text is confirmed in the
# crops saved by `montage`.  Everything within +/- `ANNOTATION_HALF_WIDTH` francs of
# it is masked out before detection.
#
# Market 5's panel also carries a genuine flat run of trades at 175, which is why
# masking cannot be done by "lowest full-width band": that band is real.
ANNOTATION_FRANCS = {1: 174.4, 2: 175.0, 3: 123.3, 4: 126.1, 5: 125.4}
ANNOTATION_HALF_WIDTH = 22.0

INK_LEVEL = 128          # grayscale threshold
LINE_MAX_THICK = 9       # a reference line is never thicker than this
DOT_MIN_THICK = 10       # a dot (or a run of dots) is never thinner than this
FULL_WIDTH_FRAC = 0.80   # inked fraction of a period's width for a "spanning" row


# ---------------------------------------------------------------- primitives


def longest_run(col: np.ndarray) -> int:
    """Length of the longest True run in a boolean vector."""
    b = np.flatnonzero(np.diff(np.concatenate(([0], col.view(np.int8), [0]))))
    return int((b[1::2] - b[0::2]).max()) if b.size else 0


def runs(col: np.ndarray) -> list[tuple[int, int]]:
    """[(start, stop)] of every True run in a boolean vector."""
    b = np.flatnonzero(np.diff(np.concatenate(([0], col.view(np.int8), [0]))))
    return list(zip(b[0::2].tolist(), b[1::2].tolist()))


def group_indices(idx, gap: int = 6) -> list[int]:
    """Collapse near-consecutive indices into their group centres."""
    idx = sorted(int(i) for i in idx)
    if not idx:
        return []
    out, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= gap:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    return [int(round(np.mean(g))) for g in out]


def load_upright(path, market: int) -> np.ndarray:
    """Grayscale array with the panel the right way up."""
    im = Image.open(path).convert("L")
    if market in ROTATE:
        im = im.rotate(ROTATE[market], expand=True)
    return np.asarray(im).astype(np.uint8)


# ---------------------------------------------------------------- calibration


def find_separators(ink: np.ndarray, market: int) -> list[int]:
    """Column positions of the period-separating vertical rules, left to right."""
    vr = np.array([longest_run(ink[:, c]) for c in range(ink.shape[1])])
    verts = group_indices(np.flatnonzero(vr > SEP_RUN_MIN[market]), gap=8)
    want = N_PERIODS[market] + 1
    if len(verts) != want:
        raise ValueError(f"market {market}: found {len(verts)} separators, want {want}")
    return verts


def _tick_rows(ink: np.ndarray, axis: int, min_len: int = 18, reach: int = 80):
    """Rows of the major (long) ticks on the left axis, with their lengths."""
    seg = ink[:, axis + 4: axis + 4 + reach]
    lens = np.zeros(seg.shape[0], int)
    for r in range(seg.shape[0]):
        row = seg[r]
        n = 0
        while n < len(row) and row[n]:
            n += 1
        lens[r] = n
    rows = np.flatnonzero(lens >= min_len)
    out = []
    for g in group_indices(rows, gap=6):
        lo, hi = g - 6, g + 7
        out.append((g, int(lens[lo:hi].max())))
    return out


def _longest_ladder(rows: list[int], tol: int = 8) -> list[int]:
    """The longest arithmetic progression hidden in a set of row positions.

    The axis carries minor ticks and the panel also sits next to body text, so the
    raw tick candidates are noisy; the major ticks are the only evenly spaced family
    with a step of a couple of hundred pixels.
    """
    rows = sorted(rows)
    best: list[int] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            step = rows[j] - rows[i]
            if step < 150:
                continue
            chain, cur = [rows[i]], rows[i]
            while True:
                target = cur + step
                near = [r for r in rows if abs(r - target) <= tol]
                if not near:
                    break
                cur = min(near, key=lambda r: abs(r - target))
                chain.append(cur)
            if len(chain) > len(best):
                best = chain
    return best


def _baseline_shear(ink: np.ndarray, verts: list[int], zero_row: int):
    """Fit the drawn 0-franc axis as a line in (col, row).

    Every panel is very slightly rotated on the page -- 0.2 to 0.3 degrees -- so a
    price read from a single pixels->francs fit taken at the left axis drifts by up
    to 2.7 francs by the right-hand end of the figure.  The zero line is drawn all
    the way across, which makes it a direct measurement of that drift: subtracting
    the fitted baseline row from every measured row removes it exactly, and the
    residual scatter about the fit (returned as `resid_sd`) bounds what is left.
    """
    xs, ys = [], []
    for c in range(verts[0], verts[-1] + 1, 7):
        rr = [t for t in runs(ink[zero_row - 25: zero_row + 26, c])
              if 2 <= t[1] - t[0] <= 9]
        if len(rr) == 1:
            s, e = rr[0]
            xs.append(c)
            ys.append(zero_row - 25 + (s + e - 1) / 2)
    xs, ys = np.array(xs), np.array(ys)
    a, b = np.polyfit(xs, ys, 1)
    return float(a), float(b), float((ys - (a * xs + b)).std()), len(xs)


def calibrate(gray: np.ndarray, market: int) -> dict:
    """Pixel geometry of one panel: separators, y-axis fit, plotting rectangle."""
    ink = gray < INK_LEVEL
    verts = find_separators(ink, market)
    axis = verts[0]
    ladder = _longest_ladder([r for r, _ in _tick_rows(ink, axis)])
    values = [TOP_TICK[market] - 50 * i for i in range(len(ladder))]
    slope, intercept = np.polyfit(ladder, values, 1)
    resid = np.abs(np.array(values) - (slope * np.array(ladder) + intercept))
    zero_row = int((0.0 - intercept) / slope)
    sh_a, sh_b, sh_sd, sh_n = _baseline_shear(ink, verts, zero_row)
    return dict(
        market=market, ink=ink, verts=verts, axis=axis,
        tick_rows=ladder, tick_values=values,
        slope=float(slope), intercept=float(intercept),
        tick_resid_max=float(resid.max()),
        shear_a=sh_a, shear_b=sh_b, shear_resid_sd=sh_sd, shear_n=sh_n,
        shear_francs=float(sh_a * (verts[-1] - verts[0]) * slope),
        # The plotting rectangle: a little above the top tick (prices can exceed it)
        # down to the zero tick.
        ytop=int(min(ladder)) - 60, ybot=int(max(ladder)) + 5,
    )


def to_francs(cal: dict, y, x=None) -> np.ndarray:
    """Price in francs of a pixel row, corrected for the panel's rotation.

    `x` is the column the row was measured at.  Omitting it gives the uncorrected
    reading, which is what the y-axis ticks themselves were fitted on.
    """
    y = np.asarray(y, float)
    if x is None:
        return cal["slope"] * y + cal["intercept"]
    base = cal["shear_a"] * np.asarray(x, float) + cal["shear_b"]
    return cal["slope"] * (y - base)


# ---------------------------------------------------------------- one period


PERIOD_PAD = 8   # the widest separator measured is 6 px off centre (market 5)


def period_window(cal: dict, period: int, pad: int = PERIOD_PAD):
    """(x0, x1) columns strictly inside the period, separators excluded.

    The separators are 2-4 px wide but market 5's are drawn with a thickened cap,
    which reaches 6 px off centre and is thick enough to survive as a dot core if
    the window is not padded past it.
    """
    v = cal["verts"]
    return v[period - 1] + pad, v[period] - pad


def thickness_at(cal: dict, x0: int, x1: int, row: int) -> np.ndarray:
    """Length of the ink run containing `row`, for every column in [x0, x1)."""
    ink = cal["ink"]
    out = np.zeros(x1 - x0, int)
    for c in range(x0, x1):
        for s, e in runs(ink[:, c]):
            if s <= row < e:
                out[c - x0] = e - s
                break
    return out


def classify_spanning_bands(cal: dict, period: int, pad: int = 5) -> list[dict]:
    """Every horizontal band that spans the period, tagged 'line' or 'dotrun'.

    A reference line and a long flat run of trades both span the panel; only their
    thickness tells them apart.
    """
    x0, x1 = period_window(cal, period, pad)
    ink = cal["ink"]
    band = ink[cal["ytop"]:cal["ybot"], x0:x1]
    frac = band.mean(1)
    rows = np.flatnonzero(frac > FULL_WIDTH_FRAC) + cal["ytop"]
    out = []
    for centre in group_indices(rows, gap=3):
        th = thickness_at(cal, x0, x1, centre)
        th = th[th > 0]
        if th.size == 0:
            continue
        med = float(np.median(th))
        out.append(dict(
            row=centre, francs=float(to_francs(cal, centre)),
            median_thick=med, p90_thick=float(np.percentile(th, 90)),
            kind="line" if med <= LINE_MAX_THICK else "dotrun",
        ))
    return out


def strip_reference_lines(cal: dict, period: int, pad: int = 5):
    """Period sub-image with the RE/PI reference lines erased.

    Only ink runs that (a) contain a row belonging to a band classified as a line
    and (b) are themselves no thicker than a line are removed.  A dot that happens
    to sit on the line is thicker than the line and so survives.
    """
    x0, x1 = period_window(cal, period, pad)
    sub = cal["ink"][cal["ytop"]:cal["ybot"], x0:x1].copy()
    bands = classify_spanning_bands(cal, period, pad)
    line_rows = set()
    for b in bands:
        if b["kind"] == "line":
            r = b["row"] - cal["ytop"]
            half = int(np.ceil(b["median_thick"] / 2)) + 1
            line_rows.update(range(r - half, r + half + 1))
    if line_rows:
        for c in range(sub.shape[1]):
            for s, e in runs(sub[:, c]):
                if (e - s) <= LINE_MAX_THICK and any(r in line_rows for r in range(s, e)):
                    sub[s:e, c] = False
    return sub, x0, cal["ytop"], bands


def grid_pitch(cal: dict, pad: int = PERIOD_PAD) -> float:
    """The horizontal spacing of consecutive trades, in pixels, for this figure.

    Trades are plotted at a fixed x increment, so the gap between the centres of
    adjacent isolated dots is the same everywhere in a panel.  Measured as the modal
    gap between narrow dot cores over all periods.
    """
    gaps = []
    for p in range(1, N_PERIODS[cal["market"]] + 1):
        sub, x0, _, _ = strip_reference_lines(cal, p, pad)
        dt = ndi.distance_transform_edt(sub)
        lab, n = ndi.label(dt >= 4.2)
        xs = []
        for i in range(1, n + 1):
            _, cx = np.nonzero(lab == i)
            if cx.max() - cx.min() <= 10:
                xs.append(cx.mean())
        xs = np.sort(xs)
        gaps += [g for g in np.diff(xs) if 7 < g < 20]
    gaps = np.asarray(gaps)
    hist, edges = np.histogram(gaps, bins=np.arange(7, 20.2, 0.2))
    mode = edges[hist.argmax()] + 0.1
    # Refine: mean of the gaps within one bin-width of the mode.
    near = gaps[np.abs(gaps - mode) < 1.2]
    return float(near.mean()) if near.size else float(mode)


DOT_CORE_DT = 5.0     # a dot's centre is >=5 px from white; a line's is <=4
TRACE_DT = 4.8        # a component containing a dot; letters/ticks never reach this
LINE_TOL = 3.0        # francs: how close a spanning band must sit to RE/PI to be one
# px: two cores this close in x are fragments of one dot, not two trades.  Chosen by
# sweeping 0-4 against the printed period means: 1 minimises both the median and the
# p90 residual (0.54 / 1.34 francs, against 0.62 / 1.44 at 0), and larger values start
# swallowing genuine adjacent trades in the flat runs of market 2.
MERGE_GAP = 1


def spanning_bands(cal: dict, period: int, pad: int = PERIOD_PAD,
                   frac: float = FULL_WIDTH_FRAC) -> list[dict]:
    """Horizontal bands that span the whole period, with their thickness profile."""
    x0, x1 = period_window(cal, period, pad)
    sub = cal["ink"][cal["ytop"]:cal["ybot"], x0:x1]
    cut = ANNOTATION_FRANCS[cal["market"]] + ANNOTATION_HALF_WIDTH
    cut_row = int((cut - cal["intercept"]) / cal["slope"]) - cal["ytop"]
    prof = sub.mean(1).copy()
    if 0 < cut_row < prof.size:
        prof[cut_row:] = 0.0
    rows = np.flatnonzero(prof > frac)
    out = []
    for g in group_indices(rows, gap=3):
        row = g + cal["ytop"]
        th = thickness_at(cal, x0, x1, row)
        th = th[th > 0]
        if th.size == 0:
            continue
        out.append(dict(row=row, francs=float(to_francs(cal, row, (x0 + x1) / 2)),
                        median_thick=float(np.median(th)),
                        thin_frac=float((th <= LINE_MAX_THICK).mean())))
    return out


def detect_trades(cal: dict, period: int, pitch: float, re_price: float,
                  pi_price: float, pad: int = PERIOD_PAD) -> dict:
    """Every trade in one period, in chronological (left-to-right) order.

    Three things share the panel with the dots and each is removed on a different
    principle:

      * the RE/PI reference lines -- a band spanning the whole period whose median
        thickness is that of a rule (<= 9 px) AND which sits within `LINE_TOL` francs
        of a price the authors would have drawn a rule at (RE, PI, or zero).  Only
        the thin ink ON those rows is erased, so a dot that happens to lie on the
        line, being thicker than the line, survives.
      * the axis ticks, the tick labels and the in-panel annotations -- ink whose
        strokes are thinner than a dot.  Any connected component whose interior never
        gets `TRACE_DT` pixels away from white contains no dot and is dropped.
      * the thin polyline joining the dots -- eroded away by taking only the cores of
        the surviving ink (distance transform >= `DOT_CORE_DT`).

    A core wider than a single dot is a run of consecutive trades at one price; it is
    split into round((w - w1) / pitch) + 1 trades, w1 being the width of a lone core
    in this period.  Those splits are reported in `run_flags` so the caller can mark
    the period's trade COUNT as uncertain even where its mean is exact.
    """
    x0, x1 = period_window(cal, period, pad)
    y0, y1 = cal["ytop"], cal["ybot"]
    sub = cal["ink"][y0:y1, x0:x1].copy()

    # The information-condition arrow and everything below it (tick labels, the
    # AVERAGE PRICE / EFFICIENCY rows, the x-axis) are not price data.
    ann = ANNOTATION_FRANCS[cal["market"]] + ANNOTATION_HALF_WIDTH
    ann_row = int((ann - cal["intercept"]) / cal["slope"]) - y0
    if 0 < ann_row < sub.shape[0]:
        sub[ann_row:, :] = False

    # Period 1 abuts the y axis, whose ticks and the axis rule itself are ink of
    # roughly a dot's thickness.  Blanking the axis column plus its tick stubs is
    # safe because no trade is ever plotted on the axis line -- the first trade of
    # every period in every figure sits at least 5 px inside it.
    if period == 1:
        lo = max(0, cal["axis"] - 6 - x0)
        hi = min(sub.shape[1], cal["axis"] + 7 - x0)
        if hi > lo:
            sub[:, lo:hi] = False

    bands = spanning_bands(cal, period, pad)
    line_rows, kept_bands = set(), []
    for b in bands:
        near = min(abs(b["francs"] - re_price), abs(b["francs"] - pi_price),
                   abs(b["francs"]))
        if b["median_thick"] <= LINE_MAX_THICK and near <= LINE_TOL:
            r = b["row"] - y0
            half = int(np.ceil(b["median_thick"] / 2)) + 1
            line_rows.update(range(r - half, r + half + 1))
        else:
            kept_bands.append(round(b["francs"], 1))
    if line_rows:
        for c in range(sub.shape[1]):
            for s, e in runs(sub[:, c]):
                if (e - s) <= LINE_MAX_THICK and any(r in line_rows
                                                     for r in range(s, e)):
                    sub[s:e, c] = False

    lab, n = ndi.label(sub, structure=np.ones((3, 3), bool))
    dt_all = ndi.distance_transform_edt(sub)
    trace = np.zeros_like(sub)
    for i in range(1, n + 1):
        m = lab == i
        if dt_all[m].max() >= TRACE_DT:
            trace |= m

    dt = ndi.distance_transform_edt(trace)
    dl, dn = ndi.label(dt >= DOT_CORE_DT)
    cores = []
    for i in range(1, dn + 1):
        ys, xs = np.nonzero(dl == i)
        cores.append(dict(w=int(xs.max() - xs.min() + 1),
                          xmin=int(xs.min()), xmax=int(xs.max()),
                          xc=float(xs.mean()),
                          area=int(len(ys)),
                          francs=float(to_francs(cal, ys.mean() + y0,
                                                 xs.mean() + x0))))
    cores.sort(key=lambda c: c["xc"])

    # A dot sitting on a steep segment of the polyline can have its core broken into
    # two pieces that overlap in x -- the same trade counted twice, at two prices a
    # few francs apart.  Cores whose x ranges overlap are one dot; merge them,
    # weighting the price by core area so the larger piece dominates.
    # Consecutive trades are a full grid pitch apart, so two cores closer than a
    # third of a pitch are always two fragments of one dot, never two trades.
    gap_tol = MERGE_GAP
    merged: list[dict] = []
    for c in cores:
        if merged and c["xmin"] <= merged[-1]["xmax"] + gap_tol:
            a, b = merged[-1], c
            wa, wb = a.get("area", 1), c.get("area", 1)
            a["francs"] = (a["francs"] * wa + b["francs"] * wb) / (wa + wb)
            a["xmax"] = max(a["xmax"], b["xmax"])
            a["xmin"] = min(a["xmin"], b["xmin"])
            a["w"] = a["xmax"] - a["xmin"] + 1
            a["xc"] = (a["xc"] * wa + b["xc"] * wb) / (wa + wb)
            a["area"] = wa + wb
        else:
            merged.append(dict(c))
    cores = merged

    lone = [c["w"] for c in cores if c["w"] <= 10]
    w1 = float(np.median(lone)) if lone else 4.0
    trades, run_flags = [], []
    for c in cores:
        k = 1 if c["w"] <= w1 + 0.6 * pitch else int(round((c["w"] - w1) / pitch)) + 1
        if k == 1:
            trades.append((c["xc"], c["francs"]))
        else:
            for j in range(k):
                trades.append((c["xmin"] + (c["w"] - 1) * j / (k - 1), c["francs"]))
            run_flags.append(f"{k}@{c['francs']:.0f}")
    trades.sort(key=lambda t: t[0])
    return dict(period=period, prices=np.array([t[1] for t in trades]),
                n=len(trades), n_cores=len(cores), w1=w1, run_flags=run_flags,
                unmatched_bands=kept_bands, trace=trace, sub=sub,
                x0=x0, y0=y0, cores=cores)


def _unused_detect_period(cal: dict, period: int, pitch: float, pad: int = 5,
                          dt_thresh: float = 4.2) -> dict:
    """Trades in one period: their prices in francs, in chronological order.

    Dots are found as the cores of the ink (points more than `dt_thresh` pixels from
    any white pixel), which excludes the thin polyline joining them.  A core wider
    than one dot is a run of trades at one price and is split into
    round((w - w1) / pitch) + 1 trades, where w1 is the width of a lone dot core.
    """
    sub, x0, y0, bands = strip_reference_lines(cal, period, pad)
    dt = ndi.distance_transform_edt(sub)
    lab, n = ndi.label(dt >= dt_thresh)
    cores = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        cores.append(dict(xmin=int(xs.min()), xmax=int(xs.max()),
                          w=int(xs.max() - xs.min() + 1),
                          xc=float(xs.mean()), yc=float(ys.mean()),
                          area=int(len(ys))))
    cores.sort(key=lambda c: c["xc"])
    lone = [c["w"] for c in cores if c["w"] <= 10]
    w1 = float(np.median(lone)) if lone else 5.0

    trades, flags = [], []
    for c in cores:
        k = 1 if c["w"] <= w1 + 0.6 * pitch else int(round((c["w"] - w1) / pitch)) + 1
        k = max(1, k)
        if k == 1:
            trades.append((c["xc"], float(to_francs(cal, c["yc"] + y0))))
        else:
            # A run: spread k trades evenly over the core and give them all the
            # run's price.  Their order within the run is their x order, which is
            # the chronological order the figure encodes.
            price = float(to_francs(cal, c["yc"] + y0))
            for j in range(k):
                trades.append((c["xmin"] + (c["w"] - 1) * j / (k - 1), price))
            flags.append(f"run{k}@{price:.0f}")
    trades.sort(key=lambda t: t[0])
    prices = np.array([t[1] for t in trades])
    return dict(period=period, prices=prices, n=len(prices), cores=cores,
                bands=bands, w1=w1, run_flags=flags, sub=sub, x0=x0, y0=y0)
