"""Numbers transcribed verbatim from the printed rows of Plott and Sunder's
Figures 2-6.

Each panel prints, along its bottom edge, the period's AVERAGE PRICE and the two
efficiency measures, and along its x-axis the realized state. `PRICES` is the
validation ground truth for the digitization: `human_trades.py` recovers the
individual trades from the plotted dots and checks that each period's recovered
mean reproduces the printed mean, without ever fitting to it.

Market m is plotted in Figure m+1.
"""

PRICES = {
    1: [239, 269, 272, 280, 289, 290, 301, 289, 310, 328, 347],
    2: [255, 251, 258, 265, 304, 337, 307, 328, 267, 327, 335],
    3: [234, 217, 189, 333, 163, 173, 364, 165, 396, 167, 177, 395],
    4: [784, 180, 154, 158, 157, 254, 161, 164, 301, 167, 323, 171, 344, 159],
    5: [165, 172, 175, 176, 175, 237, 206, 277, 240, 240, 180, 240, 296],
}
# The states printed on each figure's x-axis, for cross-checking the repo's Table 1.
FIG_STATES = {
    1: "Y Y X Y Y X Y Y Y X Y",
    2: "X X Y Y Y Y X Y X Y Y",
    3: "X Y Y X Y Y X Y X Y Y X",
    4: "X Y Y X Y X Y Y X Y X Y X Y",
    5: "Y X Z X X Y Z Z Y Y X Y Z",
}
# Per-period efficiency E and TE, transcribed from the same figures' printed rows.
EFF = {
 1: [100, 99.3, 99.3, 99.0, 99.2, 92.3, 100, 95, 96, 100, 99],
 2: [99, 98, 100, 101, 101, 100, 57, 100, 70, 100, 100],
 3: [95, 98, 79, 100, 88, 89, 100, 98, 100, 99, 100, 100],
 4: [91, 96, 93, 92, 92, 100, 95, 93, 100, 94, 100, 94, 100, 90],
 5: [91, 99, 100, 82, 94, 87, 100, 100, 100, 100, 100, 100, 100],
}
TEFF = {
 1: [100, 67, 67, 50, 97, 10, 100, 33, 95, 100, 97],
 2: [88, 75, 100, 113, 106, 100, -72, 100, -10, 100, 100],
 3: [54, 82, 13, 100, 38, 47, 100, 88, 100, 94, 100, 100],
 4: [27, 64, 50, 40, 59, 100, 72, 66, 100, 69, 100, 69, 100, 32],
 5: [37, 95, 100, -15, 65, 58, 100, 100, 100, 100, 100, 100, 100],
}

# Period 1 of market 4 is an average price of 784 in a market whose prior level is 210,
# recorded in the first period before any subject had traded. The original excludes it
# from one calculation of its own -- the note under its Table 9 reads verbatim "The data
# for period 1 of market 4 were excluded in calculating the means", which refers to the
# means of the trading-rule returns in that table, not to the price figures. Here the
# period is a no-information period and so never enters the by-side analysis in any
# case; the flag below only records it explicitly.
