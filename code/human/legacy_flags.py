"""Flags carried over from the figure-digitized record, kept only for comparison.

The earlier analysis of this experiment recovered its prices by detecting the plotted
dots in Figures 2-6 of the published article.  Two of its data-quality problems have
no counterpart in the appendix transcription, and both are recorded here so that the
new record can be scored on the OLD record's terms whenever a published result rested
on them.

MERGED_RUN_PERIODS
    Periods in which the digitizer could not resolve a run of same-priced trades into
    individual dots and inferred the count from the grid pitch.  The published
    robustness table drops these 23 periods as a sensitivity check; the appendix gives
    exact counts, so nothing here is uncertain any more, and the same 23 periods are
    dropped again only to show what that check was actually measuring.

DIGITIZER_ANOMALIES
    Periods whose digitized mean could not be reconciled with the average price the
    article prints under the panel.  Market 4's periods 7 and 8 were attributed to the
    source ("no set of trades in this panel can average the printed value"); the
    appendix reproduces both printed means to under half a franc, so the failure was
    the digitizer's.  Market 1's period 2 fails under both records and is the one
    genuine inconsistency in the original.
"""
from __future__ import annotations

MERGED_RUN_PERIODS = frozenset({
    (1, 4),
    (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),
    (3, 11), (3, 12),
    (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10),
    (4, 13), (4, 14),
})

DIGITIZER_ANOMALIES = {
    (1, 2): "mean_mismatch under both records",
    (4, 7): "source_inconsistent under the digitized record; resolved by the appendix",
    (4, 8): "source_inconsistent under the digitized record; resolved by the appendix",
}
